"""
Script para cargar datos de exoplanetas desde los datasets de NASA
al modelo Django ExoplanetData

Maneja los siguientes datasets:
- Kepler Objects of Interest (cumulative)
- K2 Planets and Candidates  
- TESS Project Candidates (TOI)
"""

import os
import sys
import django
import pandas as pd
import numpy as np
from datetime import datetime

# Configurar Django
sys.path.append('/Users/alvaroencinas/Desktop/PROY_NASA/hunting_exoplanets_ai')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core_project.settings')
django.setup()

from apps.exoplanet_ai.models import ExoplanetData


def normalize_disposition(disposition_str, source_mission):
    """
    Normaliza las disposiciones de diferentes misiones a nuestro formato estándar
    """
    if pd.isna(disposition_str) or disposition_str == '':
        return 'UNKNOWN'
    
    disposition = str(disposition_str).upper().strip()
    
    # Mapeo de disposiciones comunes
    disposition_mapping = {
        'CONFIRMED': 'CONFIRMED',
        'CANDIDATE': 'CANDIDATE',
        'FALSE POSITIVE': 'FALSE_POSITIVE',
        'FP': 'FALSE_POSITIVE',
        'FALSE_POSITIVE': 'FALSE_POSITIVE',
        'NOT TRANSIT-LIKE': 'NOT_TRANSIT',
        'NOT TRANSIT': 'NOT_TRANSIT',
        'STELLAR ECLIPSE': 'STELLAR_ECLIPSE',
        'CENTROID OFFSET': 'CENTROID_OFFSET',
        'EPHEMERIS MATCH': 'EPHEMERIS_MATCH',
        'PC': 'CANDIDATE',  # Planet Candidate (TESS)
        'APC': 'CANDIDATE',  # Ambiguous Planet Candidate (TESS)
        'CP': 'CONFIRMED',   # Confirmed Planet
    }
    
    return disposition_mapping.get(disposition, 'UNKNOWN')


def load_kepler_data(file_path):
    """
    Carga datos del dataset de Kepler Objects of Interest
    """
    print(f"Cargando datos de Kepler desde: {file_path}")
    
    # Leer CSV omitiendo comentarios
    df = pd.read_csv(file_path, comment='#', low_memory=False)
    
    print(f"Datos cargados: {len(df)} registros")
    
    # Mapeo de columnas Kepler a nuestro modelo
    kepler_objects = []
    processed_ids = set()  # Para evitar duplicados
    
    for idx, row in df.iterrows():
        try:
            object_id = str(row.get('kepid', ''))
            
            # Evitar duplicados
            if object_id in processed_ids or not object_id:
                continue
            processed_ids.add(object_id)
            
            obj = ExoplanetData(
                source_mission='KEPLER',
                object_id=object_id,
                object_name=row.get('kepoi_name', ''),
                planet_name=row.get('kepler_name', ''),
                
                # Variables principales para ML
                orbital_period=float(row['koi_period']) if pd.notna(row.get('koi_period')) else None,
                orbital_period_err=float(row['koi_period_err1']) if pd.notna(row.get('koi_period_err1')) else None,
                
                transit_duration=float(row['koi_duration']) if pd.notna(row.get('koi_duration')) else None,
                transit_duration_err=float(row['koi_duration_err1']) if pd.notna(row.get('koi_duration_err1')) else None,
                
                planetary_radius=float(row['koi_prad']) if pd.notna(row.get('koi_prad')) else None,
                planetary_radius_err=float(row['koi_prad_err1']) if pd.notna(row.get('koi_prad_err1')) else None,
                
                # Variables adicionales
                transit_epoch=float(row['koi_time0bk']) if pd.notna(row.get('koi_time0bk')) else None,
                transit_depth=float(row['koi_depth']) if pd.notna(row.get('koi_depth')) else None,
                impact_parameter=float(row['koi_impact']) if pd.notna(row.get('koi_impact')) else None,
                equilibrium_temperature=float(row['koi_teq']) if pd.notna(row.get('koi_teq')) else None,
                
                # Clasificaciones
                original_disposition=normalize_disposition(row.get('koi_disposition'), 'KEPLER'),
                disposition_score=float(row['koi_score']) if pd.notna(row.get('koi_score')) else None,
                
                # Información estelar
                stellar_temperature=float(row['koi_steff']) if pd.notna(row.get('koi_steff')) else None,
                stellar_radius=float(row['koi_srad']) if pd.notna(row.get('koi_srad')) else None,
                stellar_mass=float(row['koi_smass']) if pd.notna(row.get('koi_smass')) else None,
                
                # Metadatos
                data_source_file=os.path.basename(file_path),
                notes=row.get('koi_comment', '') if pd.notna(row.get('koi_comment')) else ''
            )
            
            # Validar que tenemos al menos las variables principales
            if obj.orbital_period and obj.transit_duration and obj.planetary_radius:
                kepler_objects.append(obj)
                
        except (ValueError, TypeError) as e:
            print(f"Error procesando fila {idx}: {e}")
            continue
    
    print(f"Objetos válidos para insertar: {len(kepler_objects)}")
    return kepler_objects


def load_tess_data(file_path):
    """
    Carga datos del dataset de TESS Project Candidates (TOI)
    """
    print(f"Cargando datos de TESS desde: {file_path}")
    
    # Leer CSV omitiendo comentarios
    df = pd.read_csv(file_path, comment='#', low_memory=False)
    
    print(f"Datos cargados: {len(df)} registros")
    
    tess_objects = []
    
    for idx, row in df.iterrows():
        try:
            obj = ExoplanetData(
                source_mission='TESS',
                object_id=str(row.get('TIC ID', '') or row.get('ticid', '')),
                object_name=f"TOI-{row.get('TOI', '')}" if pd.notna(row.get('TOI')) else '',
                planet_name=row.get('Planet Name', '') if pd.notna(row.get('Planet Name')) else '',
                
                # Variables principales para ML
                orbital_period=float(row['Period (days)']) if pd.notna(row.get('Period (days)')) else None,
                orbital_period_err=float(row['Period (days) err']) if pd.notna(row.get('Period (days) err')) else None,
                
                transit_duration=float(row['Duration (hours)']) if pd.notna(row.get('Duration (hours)')) else None,
                transit_duration_err=float(row['Duration (hours) err']) if pd.notna(row.get('Duration (hours) err')) else None,
                
                planetary_radius=float(row['Planet Radius (R_Earth)']) if pd.notna(row.get('Planet Radius (R_Earth)')) else None,
                planetary_radius_err=float(row['Planet Radius (R_Earth) err']) if pd.notna(row.get('Planet Radius (R_Earth) err')) else None,
                
                # Variables adicionales
                transit_epoch=float(row['Epoch (BJD)']) if pd.notna(row.get('Epoch (BJD)')) else None,
                transit_depth=float(row['Depth (ppm)']) if pd.notna(row.get('Depth (ppm)')) else None,
                equilibrium_temperature=float(row['Planet Equilibrium Temperature (K)']) if pd.notna(row.get('Planet Equilibrium Temperature (K)')) else None,
                
                # Clasificaciones  
                original_disposition=normalize_disposition(row.get('TFOPWG Disposition'), 'TESS'),
                
                # Información estelar
                stellar_temperature=float(row['Stellar Eff Temp (K)']) if pd.notna(row.get('Stellar Eff Temp (K)')) else None,
                stellar_radius=float(row['Stellar Radius (R_Sun)']) if pd.notna(row.get('Stellar Radius (R_Sun)')) else None,
                stellar_mass=float(row['Stellar Mass (M_Sun)']) if pd.notna(row.get('Stellar Mass (M_Sun)')) else None,
                
                # Metadatos
                data_source_file=os.path.basename(file_path),
                notes=f"TOI: {row.get('TOI', '')}" if pd.notna(row.get('TOI')) else ''
            )
            
            # Validar que tenemos al menos las variables principales
            if obj.orbital_period and obj.transit_duration and obj.planetary_radius:
                tess_objects.append(obj)
                
        except (ValueError, TypeError) as e:
            print(f"Error procesando fila {idx}: {e}")
            continue
    
    print(f"Objetos válidos para insertar: {len(tess_objects)}")
    return tess_objects


def load_k2_data(file_path):
    """
    Carga datos del dataset de K2 Planets and Candidates
    """
    print(f"Cargando datos de K2 desde: {file_path}")
    
    # Leer CSV omitiendo comentarios
    df = pd.read_csv(file_path, comment='#', low_memory=False)
    
    print(f"Datos cargados: {len(df)} registros")
    
    k2_objects = []
    
    for idx, row in df.iterrows():
        try:
            obj = ExoplanetData(
                source_mission='K2',
                object_id=str(row.get('epic_id', '') or row.get('epicid', '')),
                object_name=row.get('epic_name', '') if pd.notna(row.get('epic_name')) else '',
                planet_name=row.get('pl_name', '') if pd.notna(row.get('pl_name')) else '',
                
                # Variables principales para ML
                orbital_period=float(row['pl_orbper']) if pd.notna(row.get('pl_orbper')) else None,
                orbital_period_err=float(row['pl_orbpererr1']) if pd.notna(row.get('pl_orbpererr1')) else None,
                
                transit_duration=float(row['pl_trandur']) if pd.notna(row.get('pl_trandur')) else None,
                transit_duration_err=float(row['pl_trandurerr1']) if pd.notna(row.get('pl_trandurerr1')) else None,
                
                planetary_radius=float(row['pl_rade']) if pd.notna(row.get('pl_rade')) else None,
                planetary_radius_err=float(row['pl_radeerr1']) if pd.notna(row.get('pl_radeerr1')) else None,
                
                # Variables adicionales
                transit_epoch=float(row['pl_tranmid']) if pd.notna(row.get('pl_tranmid')) else None,
                transit_depth=float(row['pl_trandep']) if pd.notna(row.get('pl_trandep')) else None,
                equilibrium_temperature=float(row['pl_eqt']) if pd.notna(row.get('pl_eqt')) else None,
                
                # Clasificaciones
                original_disposition=normalize_disposition(row.get('pl_tranflag'), 'K2'),
                
                # Información estelar
                stellar_temperature=float(row['st_teff']) if pd.notna(row.get('st_teff')) else None,
                stellar_radius=float(row['st_rad']) if pd.notna(row.get('st_rad')) else None,
                stellar_mass=float(row['st_mass']) if pd.notna(row.get('st_mass')) else None,
                
                # Metadatos
                data_source_file=os.path.basename(file_path),
                notes=f"K2 Campaign: {row.get('pl_k2flag', '')}" if pd.notna(row.get('pl_k2flag')) else ''
            )
            
            # Validar que tenemos al menos las variables principales
            if obj.orbital_period and obj.transit_duration and obj.planetary_radius:
                k2_objects.append(obj)
                
        except (ValueError, TypeError) as e:
            print(f"Error procesando fila {idx}: {e}")
            continue
    
    print(f"Objetos válidos para insertar: {len(k2_objects)}")
    return k2_objects


def main():
    """
    Función principal para cargar todos los datasets
    """
    data_dir = '/Users/alvaroencinas/Desktop/PROY_NASA/hunting_exoplanets_ai/data'
    
    # Limpiar datos existentes
    print("Limpiando datos existentes...")
    ExoplanetData.objects.all().delete()
    
    total_loaded = 0
    
    # Cargar Kepler data
    kepler_file = os.path.join(data_dir, 'cumulative_2025.10.04_14.57.23.csv')
    if os.path.exists(kepler_file):
        kepler_objects = load_kepler_data(kepler_file)
        if kepler_objects:
            ExoplanetData.objects.bulk_create(kepler_objects, batch_size=1000, ignore_conflicts=True)
            total_loaded += len(kepler_objects)
            print(f"✅ Kepler: {len(kepler_objects)} objetos cargados")
    
    # Cargar TESS data
    tess_file = os.path.join(data_dir, 'TOI_2025.10.04_14.57.55.csv')
    if os.path.exists(tess_file):
        tess_objects = load_tess_data(tess_file)
        if tess_objects:
            ExoplanetData.objects.bulk_create(tess_objects, batch_size=1000, ignore_conflicts=True)
            total_loaded += len(tess_objects)
            print(f"✅ TESS: {len(tess_objects)} objetos cargados")
    
    # Cargar K2 data
    k2_file = os.path.join(data_dir, 'k2pandc_2025.10.04_15.00.09.csv')
    if os.path.exists(k2_file):
        k2_objects = load_k2_data(k2_file)
        if k2_objects:
            ExoplanetData.objects.bulk_create(k2_objects, batch_size=1000, ignore_conflicts=True)
            total_loaded += len(k2_objects)
            print(f"✅ K2: {len(k2_objects)} objetos cargados")
    
    print(f"\n🎉 Carga completada: {total_loaded} objetos totales en la base de datos")
    
    # Mostrar estadísticas
    print("\n📊 Estadísticas por misión:")
    for mission in ['KEPLER', 'TESS', 'K2']:
        count = ExoplanetData.objects.filter(source_mission=mission).count()
        print(f"  {mission}: {count} objetos")
    
    print("\n📊 Estadísticas por disposición:")
    for disposition, _ in ExoplanetData.DISPOSITION_CHOICES:
        count = ExoplanetData.objects.filter(original_disposition=disposition).count()
        if count > 0:
            print(f"  {disposition}: {count} objetos")


if __name__ == '__main__':
    main()