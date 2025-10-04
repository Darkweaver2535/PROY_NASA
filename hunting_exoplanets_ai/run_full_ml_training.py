#!/usr/bin/env python3
"""
Script para ejecutar el entrenamiento completo del pipeline de ML
"""

import os
import sys
import django

# Configurar Django
sys.path.append('/Users/alvaroencinas/Desktop/PROY_NASA/hunting_exoplanets_ai')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core_project.settings')
django.setup()

def run_full_ml_pipeline():
    """Ejecutar el pipeline completo de machine learning"""
    print("🚀 Iniciando entrenamiento completo de ML para exoplanetas...")
    print("="*80)
    
    try:
        from ml_preprocessing import ExoplanetDataPreprocessor
        from ml_training import ExoplanetMLTrainer
        import joblib
        
        # 1. Preprocesar datos
        print("\n🔧 FASE 1: PREPROCESAMIENTO DE DATOS")
        print("-" * 50)
        
        preprocessor = ExoplanetDataPreprocessor()
        processed_data = preprocessor.preprocess_pipeline(test_size=0.2, random_state=42)
        
        if not processed_data:
            print("❌ Error en preprocesamiento")
            return False
        
        # Guardar preprocessor
        os.makedirs('ml_models', exist_ok=True)
        preprocessor.save_preprocessor('ml_models/preprocessor.pkl')
        print("💾 Preprocessor guardado: ml_models/preprocessor.pkl")
        
        # 2. Entrenar modelos
        print("\n🤖 FASE 2: ENTRENAMIENTO DE MODELOS")
        print("-" * 50)
        
        trainer = ExoplanetMLTrainer()
        comparison_df = trainer.train_all_models(
            processed_data['X_train'],
            processed_data['X_test'], 
            processed_data['y_train'],
            processed_data['y_test'],
            processed_data['feature_names']
        )
        
        # 3. Guardar resultados
        print("\n💾 FASE 3: GUARDANDO RESULTADOS")
        print("-" * 50)
        
        # Guardar comparación de modelos
        comparison_df.to_csv('ml_models/model_comparison.csv', index=False)
        print("📊 Comparación guardada: ml_models/model_comparison.csv")
        
        # Guardar trainer completo
        joblib.dump(trainer, 'ml_models/trainer.pkl')
        print("🏆 Trainer completo guardado: ml_models/trainer.pkl")
        
        # 4. Mostrar resumen final
        print("\n🎯 RESUMEN FINAL")
        print("="*50)
        print(f"🏆 Mejor modelo: {trainer.best_model_name}")
        best_results = trainer.results[trainer.best_model_name]
        print(f"📊 Accuracy: {best_results['accuracy']:.4f}")
        print(f"📊 Precision: {best_results['precision']:.4f}")
        print(f"📊 Recall: {best_results['recall']:.4f}")
        print(f"📊 F1-Score: {best_results['f1_score']:.4f}")
        
        print("\n✅ ¡Pipeline de ML completado exitosamente!")
        return True
        
    except Exception as e:
        print(f"❌ Error en pipeline de ML: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = run_full_ml_pipeline()
    if success:
        print("\n🚀 Listo para integración con Django!")
    else:
        print("\n❌ Pipeline falló - revisar errores")