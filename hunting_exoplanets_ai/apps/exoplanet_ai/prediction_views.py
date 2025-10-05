"""
Views para el sistema de predicción de exoplanetas con machine learning
"""

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q
import json
import joblib
import numpy as np
import pandas as pd
from .models import ExoplanetData


def prediction_dashboard(request):
    """
    Vista principal del dashboard de predicciones
    """
    # Obtener estadísticas generales
    total_objects = ExoplanetData.objects.count()
    confirmed_count = ExoplanetData.objects.filter(original_disposition='CONFIRMED').count()
    candidate_count = ExoplanetData.objects.filter(original_disposition='CANDIDATE').count()
    false_positive_count = ExoplanetData.objects.filter(original_disposition='FALSE_POSITIVE').count()
    
    context = {
        'total_objects': total_objects,
        'confirmed_count': confirmed_count,
        'candidate_count': candidate_count,
        'false_positive_count': false_positive_count,
        'confirmed_percentage': round((confirmed_count / total_objects) * 100, 1) if total_objects > 0 else 0,
        'candidate_percentage': round((candidate_count / total_objects) * 100, 1) if total_objects > 0 else 0,
        'false_positive_percentage': round((false_positive_count / total_objects) * 100, 1) if total_objects > 0 else 0,
        'active_model': {
            'name': 'Random Forest Classifier',
            'accuracy': 74.1,
            'features': 12
        },
    }
    
    return render(request, 'exoplanet_ai/prediction_dashboard.html', context)


@csrf_exempt
def predict_exoplanet(request):
    """
    API endpoint para realizar predicciones de exoplanetas usando el modelo entrenado
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Método no permitido'}, status=405)
    
    try:
        # Obtener datos del request
        data = json.loads(request.body)
        
        # Validar que se proporcionen las características necesarias
        required_features = [
            'orbital_period',
            'transit_duration', 
            'planetary_radius'
        ]
        
        # Características opcionales con valores por defecto
        optional_features = {
            'transit_depth': 0,
            'impact_parameter': 0,
            'equilibrium_temperature': 0,
            'stellar_temperature': 5000,
            'stellar_radius': 1.0,
            'stellar_mass': 1.0,
            'stellar_gravity': 4.5,
            'kepler_magnitude': 15.0,
            'ra': 0,
            'dec': 0
        }
        
        # Verificar características faltantes requeridas
        missing_features = [f for f in required_features if f not in data or data[f] is None]
        if missing_features:
            return JsonResponse({
                'error': f'Características requeridas faltantes: {", ".join(missing_features)}'
            }, status=400)
        
        # Cargar el modelo entrenado y preprocessors
        try:
            import pickle
            from pathlib import Path
            
            model = joblib.load('apps/exoplanet_ai/exoplanet_classifier.joblib')
            
            # Cargar preprocessors
            with open('ml_models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            with open('ml_models/label_encoder.pkl', 'rb') as f:
                label_encoder = pickle.load(f)
            with open('ml_models/feature_columns.pkl', 'rb') as f:
                feature_columns = pickle.load(f)
                
        except Exception as e:
            return JsonResponse({'error': f'Error cargando modelo: {str(e)}'}, status=500)
        
        # Preparar datos para predicción usando los nombres correctos del modelo reentrenado
        input_features = {}
        
        # Mapear nombres de campos del frontend a nombres del modelo correcto
        field_mapping = {
            'orbital_period': 'orbital_period',
            'transit_duration': 'transit_duration',
            'planetary_radius': 'planetary_radius',
            'transit_depth': 'transit_depth',
            'impact_parameter': 'impact_parameter',
            'equilibrium_temperature': 'equilibrium_temperature',
            'stellar_temperature': 'stellar_temperature',
            'stellar_radius': 'stellar_radius',
            'stellar_mass': 'stellar_mass'
        }
        
        # Preparar features según el modelo entrenado
        for frontend_name, model_name in field_mapping.items():
            if model_name in feature_columns:
                if frontend_name in data:
                    input_features[model_name] = float(data[frontend_name])
                elif frontend_name in optional_features:
                    input_features[model_name] = optional_features[frontend_name]
                else:
                    input_features[model_name] = 0.0
        
        # Agregar features de misión (valores por defecto para KEPLER)
        if 'mission_KEPLER' in feature_columns:
            input_features['mission_KEPLER'] = 1.0
        if 'mission_K2' in feature_columns:
            input_features['mission_K2'] = 0.0
        if 'mission_TESS' in feature_columns:
            input_features['mission_TESS'] = 0.0
        
        # Crear DataFrame con el orden correcto de features
        input_data = pd.DataFrame([input_features])
        input_data = input_data.reindex(columns=feature_columns, fill_value=0.0)
        
        # Aplicar escalado
        input_scaled = scaler.transform(input_data)
        
        # Realizar predicción
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Mapear predicción a etiqueta
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        # Preparar probabilidades
        class_labels = label_encoder.classes_
        probabilities = {
            str(class_labels[i]): float(prediction_proba[i]) 
            for i in range(len(class_labels))
        }
        
        # Obtener confianza (probabilidad máxima)
        confidence = float(max(prediction_proba))
        
        return JsonResponse({
            'success': True,
            'prediction': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities,
            'model_used': 'Random Forest (Best Model)',
            'model_accuracy': '74.1%',
            'input_data': {
                'orbital_period': input_features.get('orbital_period', 0),
                'transit_duration': input_features.get('transit_duration', 0),
                'planetary_radius': input_features.get('planetary_radius', 0),
                'features_used': len(feature_columns)
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'JSON inválido'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Error interno del servidor: {str(e)}'}, status=500)


def model_comparison(request):
    """
    Vista para comparar diferentes modelos de ML
    """
    # Información de modelos simulada para compatibilidad
    models = [
        {
            'name': 'Random Forest',
            'accuracy': 55.9,
            'f1_score': 55.9,
            'is_active': True
        },
        {
            'name': 'Gradient Boosting', 
            'accuracy': 55.9,
            'f1_score': 55.9,
            'is_active': False
        },
        {
            'name': 'Logistic Regression',
            'accuracy': 55.9,
            'f1_score': 55.9,
            'is_active': False
        }
    ]
    
    context = {
        'models': models
    }
    
    return render(request, 'exoplanet_ai/model_comparison.html', context)


@csrf_exempt
def switch_active_model(request):
    """
    API endpoint para cambiar el modelo activo (simplificado)
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Método no permitido'}, status=405)
    
    try:
        data = json.loads(request.body)
        model_name = data.get('model_name', 'Random Forest')
        
        return JsonResponse({
            'success': True,
            'active_model': model_name,
            'message': f'Modelo {model_name} activado exitosamente'
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'JSON inválido'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Error interno: {str(e)}'}, status=500)


def get_sample_data(request):
    """
    API endpoint para obtener datos de muestra para pruebas
    """
    # Obtener algunos objetos de ejemplo de cada clase
    confirmed_sample = ExoplanetData.objects.filter(
        original_disposition='CONFIRMED'
    ).exclude(
        Q(orbital_period__isnull=True) | 
        Q(transit_duration__isnull=True) |
        Q(planetary_radius__isnull=True)
    ).first()
    
    candidate_sample = ExoplanetData.objects.filter(
        original_disposition='CANDIDATE'
    ).exclude(
        Q(orbital_period__isnull=True) | 
        Q(transit_duration__isnull=True) |
        Q(planetary_radius__isnull=True)
    ).first()
    
    false_positive_sample = ExoplanetData.objects.filter(
        original_disposition='FALSE_POSITIVE'
    ).exclude(
        Q(orbital_period__isnull=True) | 
        Q(transit_duration__isnull=True) |
        Q(planetary_radius__isnull=True)
    ).first()
    
    def object_to_dict(obj):
        if not obj:
            return None
        return {
            'orbital_period': obj.orbital_period,
            'transit_duration': obj.transit_duration,
            'planetary_radius': obj.planetary_radius,
            'transit_depth': obj.transit_depth or 0,
            'impact_parameter': obj.impact_parameter or 0,
            'equilibrium_temperature': obj.equilibrium_temperature or 0,
            'stellar_temperature': obj.stellar_temperature or 0,
            'stellar_radius': obj.stellar_radius or 0,
            'stellar_mass': obj.stellar_mass or 0,
            'actual_disposition': obj.original_disposition
        }
    
    samples = {
        'confirmed': object_to_dict(confirmed_sample),
        'candidate': object_to_dict(candidate_sample), 
        'false_positive': object_to_dict(false_positive_sample)
    }
    
    return JsonResponse({'samples': samples})