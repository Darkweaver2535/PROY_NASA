"""
Módulo de entrenamiento y evaluación de modelos de machine learning para clasificación de exoplanetas.

Este módulo implementa:
- Múltiples algoritmos de clasificación (Random Forest, SVM, etc.)
- Entrenamiento con validación cruzada
- Evaluación de rendimiento con métricas completas
- Selección del mejor modelo
- Serialización de modelos entrenados
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import django

# Configurar Django
sys.path.append('/Users/alvaroencinas/Desktop/PROY_NASA/hunting_exoplanets_ai')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core_project.settings')
django.setup()

from apps.exoplanet_ai.models import MLModel
from ml_preprocessing import ExoplanetDataPreprocessor


class ExoplanetMLTrainer:
    """
    Clase para entrenar y evaluar modelos de machine learning para clasificación de exoplanetas
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.label_names = ['CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE']
        
    def define_models(self):
        """
        Define los modelos de machine learning a entrenar
        """
        print("🤖 Definiendo modelos de machine learning...")
        
        self.models = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7]
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'liblinear'],
                    'penalty': ['l2']
                }
            },
            'neural_network': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        print(f"✅ {len(self.models)} modelos definidos: {list(self.models.keys())}")
    
    def train_model_with_cv(self, model_name, X_train, y_train, cv_folds=5):
        """
        Entrena un modelo con validación cruzada y optimización de hiperparámetros
        """
        print(f"\n🔄 Entrenando {model_name}...")
        
        model_config = self.models[model_name]
        base_model = model_config['model']
        param_grid = model_config['params']
        
        # Configurar validación cruzada estratificada
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Búsqueda de hiperparámetros con GridSearch
        print(f"  🔍 Optimizando hiperparámetros...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='f1_macro',  # F1-score macro para clases desbalanceadas
            n_jobs=-1,  # Usar todos los procesadores
            verbose=0
        )
        
        # Entrenar
        grid_search.fit(X_train, y_train)
        
        # Obtener mejor modelo
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_
        
        print(f"  ✅ Mejores parámetros: {best_params}")
        print(f"  📊 Score CV (F1-macro): {best_cv_score:.4f}")
        
        # Realizar validación cruzada completa con el mejor modelo
        cv_scores = cross_val_score(
            best_model, X_train, y_train,
            cv=cv, scoring='f1_macro'
        )
        
        # Guardar resultados
        self.results[model_name] = {
            'model': best_model,
            'best_params': best_params,
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'best_cv_score': best_cv_score,
            'cv_scores': cv_scores
        }
        
        print(f"  🎯 CV Score final: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return best_model
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """
        Evalúa un modelo entrenado con el conjunto de test
        """
        print(f"\n📊 Evaluando {model_name} en conjunto de test...")
        
        # Realizar predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        
        # AUC para clasificación multiclase (one-vs-rest)
        auc = None
        if y_pred_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            except:
                auc = None
        
        # Reporte de clasificación detallado
        class_report = classification_report(
            y_test, y_pred,
            target_names=self.label_names,
            output_dict=True
        )
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Guardar métricas
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        # Actualizar resultados
        self.results[model_name].update(metrics)
        
        print(f"  📈 Accuracy: {accuracy:.4f}")
        print(f"  📈 Precision (macro): {precision:.4f}")
        print(f"  📈 Recall (macro): {recall:.4f}")
        print(f"  📈 F1-Score (macro): {f1:.4f}")
        if auc:
            print(f"  📈 AUC (macro): {auc:.4f}")
        
        return metrics
    
    def print_classification_report(self, model_name):
        """
        Imprime el reporte de clasificación detallado
        """
        print(f"\n📋 Reporte de clasificación para {model_name}:")
        print("=" * 60)
        
        report = self.results[model_name]['classification_report']
        
        # Imprimir métricas por clase
        for class_name in self.label_names:
            if class_name.lower() in report:
                metrics = report[class_name.lower()]
            else:
                # Buscar por índice numérico
                class_idx = self.label_names.index(class_name)
                metrics = report.get(str(class_idx), {})
            
            if metrics:
                print(f"{class_name:15} - Precision: {metrics.get('precision', 0):.3f}, "
                      f"Recall: {metrics.get('recall', 0):.3f}, "
                      f"F1: {metrics.get('f1-score', 0):.3f}, "
                      f"Support: {metrics.get('support', 0)}")
        
        # Métricas macro
        macro_avg = report.get('macro avg', {})
        print(f"\n{'MACRO AVG':15} - Precision: {macro_avg.get('precision', 0):.3f}, "
              f"Recall: {macro_avg.get('recall', 0):.3f}, "
              f"F1: {macro_avg.get('f1-score', 0):.3f}")
    
    def plot_confusion_matrix(self, model_name, save_path=None):
        """
        Visualiza la matriz de confusión
        """
        conf_matrix = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names
        )
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Matriz de confusión guardada: {save_path}")
        
        plt.show()
    
    def compare_models(self):
        """
        Compara todos los modelos entrenados y selecciona el mejor
        """
        print("\n🏆 Comparando modelos entrenados...")
        print("=" * 80)
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'CV Score': results['cv_score_mean'],
                'CV Std': results['cv_score_std'],
                'Test Accuracy': results['accuracy'],
                'Test Precision': results['precision'],
                'Test Recall': results['recall'],
                'Test F1': results['f1_score'],
                'Test AUC': results.get('auc', 'N/A')
            })
        
        # Crear DataFrame para comparación
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test F1', ascending=False)
        
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Seleccionar mejor modelo basado en F1-score
        best_model_name = comparison_df.iloc[0]['Model']
        self.best_model_name = best_model_name
        self.best_model = self.results[best_model_name]['model']
        
        print(f"\n🥇 Mejor modelo: {best_model_name}")
        print(f"   F1-Score: {comparison_df.iloc[0]['Test F1']:.4f}")
        
        return comparison_df
    
    def save_model_to_django(self, model_name):
        """
        Guarda el modelo en la base de datos Django
        """
        print(f"\n💾 Guardando {model_name} en base de datos...")
        
        results = self.results[model_name]
        model = results['model']
        
        # Crear directorio para modelos si no existe
        models_dir = 'ml_models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Rutas de archivos
        model_filename = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        
        # Guardar modelo serializado
        joblib.dump(model, model_path)
        
        # Mapeo de nombres de algoritmos
        algorithm_mapping = {
            'random_forest': 'RANDOM_FOREST',
            'gradient_boosting': 'GRADIENT_BOOSTING', 
            'svm': 'SVM',
            'logistic_regression': 'LOGISTIC_REGRESSION',
            'neural_network': 'NEURAL_NETWORK'
        }
        
        # Crear registro en Django
        ml_model = MLModel.objects.create(
            name=f"Exoplanet Classifier - {model_name.replace('_', ' ').title()}",
            algorithm=algorithm_mapping.get(model_name, 'OTHER'),
            accuracy=results['accuracy'],
            precision=results['precision'],
            recall=results['recall'],
            f1_score=results['f1_score'],
            training_size=len(results.get('X_train', [])),
            test_size=len(results.get('X_test', [])),
            features_used=list(results.get('feature_names', [])),
            hyperparameters=results['best_params'],
            model_file_path=model_path,
            is_active=(model_name == self.best_model_name),
            notes=f"Modelo entrenado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
                  f"CV Score: {results['cv_score_mean']:.4f}"
        )
        
        print(f"✅ Modelo guardado con ID: {ml_model.id}")
        return ml_model
    
    def train_all_models(self, X_train, X_test, y_train, y_test, feature_names):
        """
        Entrena y evalúa todos los modelos definidos
        """
        print("🚀 Iniciando entrenamiento de todos los modelos...")
        
        # Definir modelos
        self.define_models()
        
        # Entrenar cada modelo
        for model_name in self.models.keys():
            # Entrenar con validación cruzada
            trained_model = self.train_model_with_cv(model_name, X_train, y_train)
            
            # Evaluar en conjunto de test
            self.evaluate_model(model_name, trained_model, X_test, y_test)
            
            # Imprimir reporte detallado
            self.print_classification_report(model_name)
            
            # Guardar nombres de características para referencia
            self.results[model_name]['feature_names'] = feature_names
            self.results[model_name]['X_train'] = X_train
            self.results[model_name]['X_test'] = X_test
        
        # Comparar modelos y seleccionar el mejor
        comparison_df = self.compare_models()
        
        # Guardar todos los modelos en Django
        print("\n💾 Guardando modelos en base de datos...")
        for model_name in self.models.keys():
            self.save_model_to_django(model_name)
        
        print("\n🎉 ¡Entrenamiento completado!")
        return comparison_df


def main():
    """
    Función principal para ejecutar el pipeline completo de ML
    """
    print("🌟 Iniciando pipeline de Machine Learning para exoplanetas...")
    
    # 1. Preprocesar datos
    print("\n" + "="*60)
    print("FASE 1: PREPROCESAMIENTO DE DATOS")
    print("="*60)
    
    preprocessor = ExoplanetDataPreprocessor()
    data = preprocessor.preprocess_pipeline(test_size=0.2, random_state=42)
    
    # Guardar preprocessor
    os.makedirs('ml_models', exist_ok=True)
    preprocessor.save_preprocessor('ml_models/preprocessor.pkl')
    
    # 2. Entrenar modelos
    print("\n" + "="*60)
    print("FASE 2: ENTRENAMIENTO DE MODELOS")
    print("="*60)
    
    trainer = ExoplanetMLTrainer()
    comparison_df = trainer.train_all_models(
        data['X_train'],
        data['X_test'], 
        data['y_train'],
        data['y_test'],
        data['feature_names']
    )
    
    # 3. Guardar resultados
    print("\n" + "="*60)
    print("FASE 3: GUARDANDO RESULTADOS")
    print("="*60)
    
    # Guardar comparación de modelos
    comparison_df.to_csv('ml_models/model_comparison.csv', index=False)
    print("📊 Comparación de modelos guardada: ml_models/model_comparison.csv")
    
    # Guardar trainer completo
    joblib.dump(trainer, 'ml_models/trainer.pkl')
    print("💾 Trainer completo guardado: ml_models/trainer.pkl")
    
    print(f"\n🏆 Mejor modelo seleccionado: {trainer.best_model_name}")
    print("✅ Pipeline de ML completado exitosamente!")


if __name__ == '__main__':
    main()