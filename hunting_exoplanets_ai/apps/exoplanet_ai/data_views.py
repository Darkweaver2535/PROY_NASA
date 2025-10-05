"""
Views avanzadas para la gestión y carga de datos de exoplanetas
"""

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import pandas as pd
import json
import os
import tempfile
from datetime import datetime
from .models import ExoplanetData, MLModel
from .prediction_views import predict_exoplanet


def data_management_dashboard(request):
    """
    Dashboard para gestión de datos y carga de archivos
    """
    # Estadísticas actuales
    total_objects = ExoplanetData.objects.count()
    recent_objects = ExoplanetData.objects.filter(
        created_at__gte=datetime.now().replace(hour=0, minute=0, second=0)
    ).count()
    
    # Distribución por misión
    missions_stats = {}
    for mission in ['Kepler', 'K2', 'TESS']:
        count = ExoplanetData.objects.filter(source_mission=mission).count()
        missions_stats[mission] = count
    
    # Distribución por disposición
    disposition_stats = {}
    for disp in ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']:
        count = ExoplanetData.objects.filter(original_disposition=disp).count()
        disposition_stats[disp] = count
    
    context = {
        'total_objects': total_objects,
        'recent_objects': recent_objects,
        'missions_stats': missions_stats,
        'disposition_stats': disposition_stats,
    }
    
    return render(request, 'exoplanet_ai/data_management.html', context)


@csrf_exempt
def upload_csv_data(request):
    """
    API endpoint para cargar datos desde archivos CSV
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Método no permitido'}, status=405)
    
    if 'csv_file' not in request.FILES:
        return JsonResponse({'error': 'No se proporcionó archivo CSV'}, status=400)
    
    csv_file = request.FILES['csv_file']
    
    # Validar tipo de archivo
    if not csv_file.name.endswith('.csv'):
        return JsonResponse({'error': 'El archivo debe ser de tipo CSV'}, status=400)
    
    try:
        # Leer el archivo CSV
        df = pd.read_csv(csv_file)
        
        # Validar columnas requeridas
        required_columns = [
            'orbital_period', 'transit_duration', 'planetary_radius'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return JsonResponse({
                'error': f'Columnas faltantes: {", ".join(missing_columns)}',
                'available_columns': list(df.columns)
            }, status=400)
        
        # Procesar y validar datos
        processed_data = process_csv_data(df)
        
        if processed_data['errors']:
            return JsonResponse({
                'error': 'Errores en validación de datos',
                'validation_errors': processed_data['errors'],
                'valid_rows': processed_data['valid_count'],
                'total_rows': len(df)
            }, status=400)
        
        # Guardar datos válidos
        saved_count = save_csv_data(processed_data['data'])
        
        return JsonResponse({
            'success': True,
            'message': f'Datos cargados exitosamente',
            'saved_objects': saved_count,
            'total_processed': len(df),
            'valid_objects': processed_data['valid_count']
        })
        
    except pd.errors.EmptyDataError:
        return JsonResponse({'error': 'El archivo CSV está vacío'}, status=400)
    except pd.errors.ParserError as e:
        return JsonResponse({'error': f'Error parseando CSV: {str(e)}'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Error interno: {str(e)}'}, status=500)


def process_csv_data(df):
    """
    Procesa y valida los datos del CSV
    """
    processed_data = []
    errors = []
    valid_count = 0
    
    # Mapeo de columnas opcionales
    column_mapping = {
        'orbital_period': 'orbital_period',
        'transit_duration': 'transit_duration', 
        'planetary_radius': 'planetary_radius',
        'transit_depth': 'transit_depth',
        'impact_parameter': 'impact_parameter',
        'equilibrium_temperature': 'equilibrium_temperature',
        'stellar_temperature': 'stellar_temperature',
        'stellar_radius': 'stellar_radius',
        'stellar_mass': 'stellar_mass',
        'disposition': 'original_disposition',
        'mission': 'source_mission',
        'kepoi_name': 'kepoi_name',
        'kepler_name': 'kepler_name'
    }
    
    for index, row in df.iterrows():
        try:
            # Extraer y validar datos requeridos
            orbital_period = float(row['orbital_period'])
            transit_duration = float(row['transit_duration'])
            planetary_radius = float(row['planetary_radius'])
            
            # Validaciones básicas
            if orbital_period <= 0:
                errors.append(f"Fila {index + 1}: Período orbital debe ser positivo")
                continue
                
            if transit_duration <= 0:
                errors.append(f"Fila {index + 1}: Duración del tránsito debe ser positiva")
                continue
                
            if planetary_radius <= 0:
                errors.append(f"Fila {index + 1}: Radio planetario debe ser positivo")
                continue
            
            # Crear objeto de datos
            data_obj = {
                'orbital_period': orbital_period,
                'transit_duration': transit_duration,
                'planetary_radius': planetary_radius,
                'source_mission': row.get('mission', 'Custom'),
                'kepoi_name': row.get('kepoi_name', f'CUSTOM-{index + 1}'),
                'kepler_name': row.get('kepler_name', ''),
                'original_disposition': row.get('disposition', 'CANDIDATE')
            }
            
            # Agregar campos opcionales si están presentes
            optional_fields = [
                'transit_depth', 'impact_parameter', 'equilibrium_temperature',
                'stellar_temperature', 'stellar_radius', 'stellar_mass'
            ]
            
            for field in optional_fields:
                if field in row and pd.notna(row[field]):
                    try:
                        data_obj[field] = float(row[field])
                    except (ValueError, TypeError):
                        # Si no se puede convertir, usar None
                        data_obj[field] = None
                else:
                    data_obj[field] = None
            
            processed_data.append(data_obj)
            valid_count += 1
            
        except (ValueError, TypeError, KeyError) as e:
            errors.append(f"Fila {index + 1}: Error de formato - {str(e)}")
            continue
    
    return {
        'data': processed_data,
        'errors': errors,
        'valid_count': valid_count
    }


def save_csv_data(data_list):
    """
    Guarda los datos procesados en la base de datos
    """
    objects_to_create = []
    
    for data in data_list:
        # Verificar si el objeto ya existe
        existing = ExoplanetData.objects.filter(
            kepoi_name=data['kepoi_name']
        ).first()
        
        if not existing:
            obj = ExoplanetData(**data)
            objects_to_create.append(obj)
    
    # Crear objetos en lote
    created_objects = ExoplanetData.objects.bulk_create(
        objects_to_create, 
        ignore_conflicts=True
    )
    
    return len(created_objects)


def validate_data_preview(request):
    """
    Vista para previsualizar y validar datos antes de cargar
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Método no permitido'}, status=405)
    
    if 'csv_file' not in request.FILES:
        return JsonResponse({'error': 'No se proporcionó archivo CSV'}, status=400)
    
    csv_file = request.FILES['csv_file']
    
    try:
        # Leer solo las primeras 10 filas para preview
        df = pd.read_csv(csv_file, nrows=10)
        
        # Información sobre el archivo
        file_info = {
            'filename': csv_file.name,
            'size': csv_file.size,
            'columns': list(df.columns),
            'total_rows': len(df),
            'preview_data': df.to_dict('records')
        }
        
        # Verificar columnas requeridas
        required_columns = ['orbital_period', 'transit_duration', 'planetary_radius']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        file_info['missing_required_columns'] = missing_columns
        file_info['is_valid'] = len(missing_columns) == 0
        
        # Sugerencias de mapeo de columnas
        suggestions = suggest_column_mapping(df.columns)
        file_info['column_suggestions'] = suggestions
        
        return JsonResponse({
            'success': True,
            'file_info': file_info
        })
        
    except Exception as e:
        return JsonResponse({'error': f'Error procesando archivo: {str(e)}'}, status=400)


def suggest_column_mapping(columns):
    """
    Sugiere mapeo de columnas basado en nombres comunes
    """
    suggestions = {}
    
    # Mapeos comunes de nombres de columnas
    mapping_patterns = {
        'orbital_period': ['period', 'orbital_period', 'koi_period', 'pl_orbper'],
        'transit_duration': ['duration', 'transit_duration', 'koi_duration', 'pl_trandur'],
        'planetary_radius': ['radius', 'planetary_radius', 'koi_prad', 'pl_radj', 'pl_rade'],
        'transit_depth': ['depth', 'transit_depth', 'koi_depth', 'pl_trandep'],
        'stellar_temperature': ['teff', 'stellar_temp', 'koi_steff', 'st_teff'],
        'stellar_radius': ['stellar_radius', 'koi_srad', 'st_rad'],
        'stellar_mass': ['stellar_mass', 'koi_smass', 'st_mass'],
        'disposition': ['disposition', 'koi_disposition', 'pl_controv_flag']
    }
    
    for target_col, patterns in mapping_patterns.items():
        for col in columns:
            for pattern in patterns:
                if pattern.lower() in col.lower():
                    suggestions[target_col] = col
                    break
            if target_col in suggestions:
                break
    
    return suggestions


@csrf_exempt
def batch_predict_csv(request):
    """
    API endpoint para hacer predicciones en lote sobre datos CSV
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Método no permitido'}, status=405)
    
    if 'csv_file' not in request.FILES:
        return JsonResponse({'error': 'No se proporcionó archivo CSV'}, status=400)
    
    csv_file = request.FILES['csv_file']
    
    try:
        # Leer CSV
        df = pd.read_csv(csv_file)
        
        # Validar columnas requeridas para predicción
        required_features = [
            'orbital_period', 'transit_duration', 'planetary_radius',
            'transit_depth', 'impact_parameter', 'equilibrium_temperature',
            'stellar_temperature', 'stellar_radius', 'stellar_mass'
        ]
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            return JsonResponse({
                'error': f'Características faltantes para predicción: {", ".join(missing_features)}'
            }, status=400)
        
        # Realizar predicciones en lote
        predictions = []
        for index, row in df.iterrows():
            try:
                # Preparar datos para predicción
                data = {feature: float(row[feature]) for feature in required_features}
                
                # Simular llamada a la API de predicción
                # En una implementación real, usarías el modelo directamente
                from django.test import RequestFactory
                factory = RequestFactory()
                pred_request = factory.post('/api/predict/', json.dumps(data), content_type='application/json')
                
                # Aquí llamarías a predict_exoplanet directamente
                # result = predict_exoplanet(pred_request)
                
                # Por simplicidad, agregamos una predicción simulada
                predictions.append({
                    'row_index': index,
                    'prediction': 'CANDIDATE',  # Esto se calcularia realmente
                    'confidence': 0.75,
                    'input_data': data
                })
                
            except Exception as e:
                predictions.append({
                    'row_index': index,
                    'error': str(e),
                    'input_data': dict(row)
                })
        
        return JsonResponse({
            'success': True,
            'predictions': predictions,
            'total_processed': len(df)
        })
        
    except Exception as e:
        return JsonResponse({'error': f'Error procesando archivo: {str(e)}'}, status=500)


def download_sample_csv(request):
    """
    Descarga un archivo CSV de ejemplo con el formato correcto
    """
    import io
    
    # Datos de ejemplo
    sample_data = {
        'orbital_period': [3.52, 85.23, 1.64, 9.74],
        'transit_duration': [2.83, 4.12, 1.98, 3.45],
        'planetary_radius': [1.27, 2.35, 0.89, 1.78],
        'transit_depth': [1243, 2456, 892, 1567],
        'impact_parameter': [0.45, 0.23, 0.67, 0.34],
        'equilibrium_temperature': [1204, 456, 1789, 923],
        'stellar_temperature': [5777, 4567, 6234, 5234],
        'stellar_radius': [1.0, 1.34, 0.87, 1.12],
        'stellar_mass': [1.0, 1.23, 0.78, 0.95],
        'disposition': ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE', 'CANDIDATE'],
        'mission': ['Kepler', 'K2', 'TESS', 'Kepler'],
        'kepoi_name': ['K00001.01', 'K00002.01', 'K00003.01', 'K00004.01'],
        'kepler_name': ['Kepler-1b', '', '', 'Kepler-4b']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Crear respuesta CSV
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="exoplanet_sample.csv"'
    
    df.to_csv(response, index=False)
    return response