#!/usr/bin/env python3
"""
Script para ejecutar el pipeline de machine learning para exoplanetas
"""

import os
import sys
import django

# Configurar Django
sys.path.append('/Users/alvaroencinas/Desktop/PROY_NASA/hunting_exoplanets_ai')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core_project.settings')
django.setup()

def test_preprocessing():
    """Probar el módulo de preprocesamiento"""
    print("🧪 Probando el módulo de preprocesamiento...")
    
    try:
        from ml_preprocessing import ExoplanetDataPreprocessor
        
        # Crear instancia del preprocesador
        preprocessor = ExoplanetDataPreprocessor()
        
        # Cargar datos
        print("📥 Cargando datos desde Django...")
        data = preprocessor.load_data_from_django()
        print(f"✅ Datos cargados: {len(data)} registros")
        
        if len(data) > 0:
            print(f"📊 Características: {list(data.columns)}")
            print(f"📈 Distribución de disposiciones:")
            print(data['original_disposition'].value_counts())
            
            # Probar limpieza de datos
            print("\n🧹 Probando limpieza de datos...")
            clean_data = preprocessor.clean_data(data)
            print(f"✅ Datos después de limpieza: {len(clean_data)} registros")
            
            # Probar codificación de target
            print("\n🎯 Probando codificación de target...")
            encoded_target, valid_mask = preprocessor.encode_target(clean_data['original_disposition'])
            print(f"✅ Target codificado: {encoded_target.value_counts().to_dict()}")
            print(f"✅ Máscara válida: {valid_mask.sum()} registros válidos de {len(valid_mask)}")
            
            return True
        else:
            print("❌ No hay datos para procesar")
            return False
            
    except Exception as e:
        print(f"❌ Error en preprocesamiento: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_ml():
    """Probar entrenamiento básico de ML"""
    print("\n🤖 Probando entrenamiento básico de ML...")
    
    try:
        from ml_preprocessing import ExoplanetDataPreprocessor
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        
        # Preprocesar datos
        preprocessor = ExoplanetDataPreprocessor()
        processed_data = preprocessor.preprocess_pipeline(test_size=0.3, random_state=42)
        
        if processed_data:
            X_train = processed_data['X_train']
            X_test = processed_data['X_test']
            y_train = processed_data['y_train']
            y_test = processed_data['y_test']
            
            print(f"📊 Conjunto de entrenamiento: {X_train.shape}")
            print(f"📊 Conjunto de prueba: {X_test.shape}")
            
            # Entrenar modelo simple
            print("🌳 Entrenando Random Forest...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            # Evaluar
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ Accuracy: {accuracy:.4f}")
            print("\n📋 Reporte de clasificación:")
            print(classification_report(y_test, y_pred, target_names=['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']))
            
            return True
        else:
            print("❌ No se pudieron procesar los datos")
            return False
            
    except Exception as e:
        print(f"❌ Error en entrenamiento ML: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("🌟 Iniciando pruebas del pipeline de ML para exoplanetas...")
    print("="*60)
    
    # Probar preprocesamiento
    preprocessing_ok = test_preprocessing()
    
    if preprocessing_ok:
        # Probar ML básico
        ml_ok = test_basic_ml()
        
        if ml_ok:
            print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
            print("✅ El pipeline de ML está listo para uso completo")
        else:
            print("\n⚠️ Preprocesamiento OK, pero falló ML")
    else:
        print("\n❌ Falló el preprocesamiento")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()