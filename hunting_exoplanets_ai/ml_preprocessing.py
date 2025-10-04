"""
Módulo de preprocesamiento de datos para el pipeline de ML de exoplanetas.

Este módulo maneja:
- Limpieza y preparación de datos de NASA
- Codificación de etiquetas de disposición
- Estandarización y normalización de características
- División de datos para entrenamiento y testing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import os
import sys
import django

# Configurar Django
sys.path.append('/Users/alvaroencinas/Desktop/PROY_NASA/hunting_exoplanets_ai')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core_project.settings')
django.setup()

from apps.exoplanet_ai.models import ExoplanetData


class ExoplanetDataPreprocessor:
    """
    Clase para preprocesar datos de exoplanetas para machine learning
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = [
            'orbital_period',
            'transit_duration', 
            'planetary_radius',
            'transit_depth',
            'impact_parameter',
            'equilibrium_temperature',
            'stellar_temperature',
            'stellar_radius',
            'stellar_mass'
        ]
        self.target_column = 'original_disposition'
        
    def load_data_from_django(self):
        """
        Carga datos desde el modelo Django ExoplanetData
        """
        print("🔄 Cargando datos desde Django...")
        
        # Obtener todos los objetos con datos completos
        queryset = ExoplanetData.objects.filter(
            orbital_period__isnull=False,
            transit_duration__isnull=False,
            planetary_radius__isnull=False,
            original_disposition__in=['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']
        )
        
        # Convertir a DataFrame
        data = []
        for obj in queryset:
            features = obj.get_features_for_ml()
            features[self.target_column] = obj.original_disposition
            features['id'] = obj.id
            data.append(features)
        
        df = pd.DataFrame(data)
        print(f"✅ Datos cargados: {len(df)} objetos")
        print(f"📊 Distribución de clases:")
        print(df[self.target_column].value_counts())
        
        return df
    
    def clean_data(self, df):
        """
        Limpia y prepara los datos para machine learning
        """
        print("🧹 Limpiando datos...")
        
        # Crear copia para no modificar el original
        df_clean = df.copy()
        
        # 1. Manejar valores extremos (outliers)
        for col in self.feature_columns:
            if col in df_clean.columns:
                # Calcular Q1 y Q3
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Definir límites para outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Contar outliers antes
                outliers_before = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                
                # Clipear outliers (mantenerlos dentro de los límites)
                df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                
                print(f"  {col}: {outliers_before} outliers ajustados")
        
        # 2. Crear características derivadas
        if 'orbital_period' in df_clean.columns and 'planetary_radius' in df_clean.columns:
            # Densidad relativa (aproximada)
            df_clean['period_radius_ratio'] = df_clean['orbital_period'] / df_clean['planetary_radius']
        
        if 'stellar_temperature' in df_clean.columns and 'equilibrium_temperature' in df_clean.columns:
            # Ratio de temperaturas
            df_clean['temp_ratio'] = df_clean['equilibrium_temperature'] / df_clean['stellar_temperature']
        
        # 3. Validar rangos físicos realistas
        # Período orbital: debe ser positivo y menor a 10,000 días
        if 'orbital_period' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['orbital_period'] > 0) & 
                (df_clean['orbital_period'] < 10000)
            ]
        
        # Radio planetario: entre 0.1 y 100 radios terrestres
        if 'planetary_radius' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['planetary_radius'] > 0.1) & 
                (df_clean['planetary_radius'] < 100)
            ]
        
        print(f"✅ Datos después de limpieza: {len(df_clean)} objetos")
        return df_clean
    
    def prepare_features_and_target(self, df):
        """
        Prepara las características y el target para machine learning
        """
        print("🎯 Preparando características y target...")
        
        # Seleccionar solo características disponibles
        available_features = [col for col in self.feature_columns if col in df.columns]
        print(f"📋 Características disponibles: {available_features}")
        
        # Extraer características
        X = df[available_features].copy()
        
        # Extraer target
        y = df[self.target_column].copy()
        
        # Mostrar estadísticas de características
        print(f"📊 Estadísticas de características:")
        print(X.describe())
        
        return X, y, available_features
    
    def handle_missing_values(self, X):
        """
        Maneja valores faltantes en las características
        """
        print("🔧 Manejando valores faltantes...")
        
        # Verificar valores faltantes
        missing_info = X.isnull().sum()
        if missing_info.sum() > 0:
            print(f"⚠️  Valores faltantes encontrados:")
            for col, count in missing_info.items():
                if count > 0:
                    print(f"  {col}: {count} valores ({count/len(X)*100:.1f}%)")
            
            # Imputar valores faltantes
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            print("✅ Valores faltantes imputados con la mediana")
            return X_imputed
        else:
            print("✅ No se encontraron valores faltantes")
            return X
    
    def encode_target(self, y):
        """
        Codifica las etiquetas de target a formato numérico
        """
        print("🏷️  Codificando etiquetas de target...")
        
        # Mapeo manual para control total
        label_mapping = {
            'CONFIRMED': 0,     # Exoplaneta confirmado
            'CANDIDATE': 1,     # Candidato a planeta
            'FALSE_POSITIVE': 2 # Falso positivo
        }
        
        # Aplicar mapeo
        y_encoded = y.map(label_mapping)
        
        # Verificar que no hay valores perdidos
        if y_encoded.isnull().sum() > 0:
            print("⚠️  Etiquetas no reconocidas encontradas:")
            unrecognized = y[y_encoded.isnull()].unique()
            print(f"  {unrecognized}")
            # Eliminar filas con etiquetas no reconocidas
            valid_mask = ~y_encoded.isnull()
            y_encoded = y_encoded[valid_mask]
            print(f"✅ Eliminadas {(~valid_mask).sum()} filas con etiquetas no válidas")
            return y_encoded, valid_mask
        
        print(f"✅ Etiquetas codificadas:")
        for label, code in label_mapping.items():
            count = (y_encoded == code).sum()
            print(f"  {label} ({code}): {count} objetos")
        
        return y_encoded, pd.Series(True, index=y.index)
    
    def scale_features(self, X_train, X_test=None):
        """
        Estandariza las características usando StandardScaler
        """
        print("📏 Estandarizando características...")
        
        # Ajustar el scaler solo con datos de entrenamiento
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index
        )
        
        print(f"✅ Características de entrenamiento estandarizadas: {X_train_scaled.shape}")
        
        if X_test is not None:
            # Aplicar la misma transformación a los datos de test
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(
                X_test_scaled,
                columns=X_test.columns,
                index=X_test.index
            )
            print(f"✅ Características de test estandarizadas: {X_test_scaled.shape}")
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Divide los datos en conjuntos de entrenamiento y test
        """
        print(f"✂️  Dividiendo datos (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Mantener proporciones de clases
        )
        
        print(f"📊 División completada:")
        print(f"  Entrenamiento: {X_train.shape[0]} objetos")
        print(f"  Test: {X_test.shape[0]} objetos")
        
        # Mostrar distribución de clases
        print(f"📊 Distribución en entrenamiento:")
        print(y_train.value_counts().sort_index())
        print(f"📊 Distribución en test:")
        print(y_test.value_counts().sort_index())
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, test_size=0.2, random_state=42):
        """
        Pipeline completo de preprocesamiento
        """
        print("🚀 Iniciando pipeline de preprocesamiento...")
        
        # 1. Cargar datos
        df = self.load_data_from_django()
        
        # 2. Limpiar datos
        df_clean = self.clean_data(df)
        
        # 3. Preparar características y target
        X, y, feature_names = self.prepare_features_and_target(df_clean)
        
        # 4. Codificar target
        y_encoded, valid_mask = self.encode_target(y)
        
        # Aplicar máscara válida también a X
        X = X[valid_mask]
        
        # 5. Manejar valores faltantes
        X_clean = self.handle_missing_values(X)
        
        # 6. Dividir datos
        X_train, X_test, y_train, y_test = self.split_data(
            X_clean, y_encoded, test_size, random_state
        )
        
        # 7. Estandarizar características
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("🎉 Pipeline de preprocesamiento completado!")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'original_data': df_clean
        }
    
    def save_preprocessor(self, filepath):
        """
        Guarda el preprocessor entrenado
        """
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'imputer': self.imputer,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        joblib.dump(preprocessor_data, filepath)
        print(f"💾 Preprocessor guardado en: {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath):
        """
        Carga un preprocessor previamente entrenado
        """
        data = joblib.load(filepath)
        
        preprocessor = cls()
        preprocessor.scaler = data['scaler']
        preprocessor.label_encoder = data['label_encoder']
        preprocessor.imputer = data['imputer']
        preprocessor.feature_columns = data['feature_columns']
        preprocessor.target_column = data['target_column']
        
        print(f"📥 Preprocessor cargado desde: {filepath}")
        return preprocessor


def main():
    """
    Función principal para probar el preprocesamiento
    """
    print("🌟 Probando preprocesamiento de datos de exoplanetas...")
    
    # Crear preprocessor
    preprocessor = ExoplanetDataPreprocessor()
    
    # Ejecutar pipeline
    data = preprocessor.preprocess_pipeline()
    
    print(f"\n📋 Resumen final:")
    print(f"  Características de entrenamiento: {data['X_train'].shape}")
    print(f"  Características de test: {data['X_test'].shape}")
    print(f"  Características disponibles: {len(data['feature_names'])}")
    print(f"  Características: {data['feature_names']}")
    
    # Guardar preprocessor
    os.makedirs('ml_models', exist_ok=True)
    preprocessor.save_preprocessor('ml_models/preprocessor.pkl')
    
    print("\n✅ Preprocesamiento completado exitosamente!")


if __name__ == '__main__':
    main()