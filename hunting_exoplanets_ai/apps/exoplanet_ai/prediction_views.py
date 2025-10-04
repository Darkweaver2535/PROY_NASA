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
from .models import ExoplanetData, MLModel


def prediction_dashboard(request):
    """
    Vista principal del dashboard de predicciones
    """
    # Obtener estadísticas generales
    total_objects = ExoplanetData.objects.count()
    confirmed_count = ExoplanetData.objects.filter(original_disposition='CONFIRMED').count()
    candidate_count = ExoplanetData.objects.filter(original_disposition='CANDIDATE').count()
    false_positive_count = ExoplanetData.objects.filter(original_disposition='FALSE_POSITIVE').count()
    
    # Obtener modelo activo
    active_model = MLModel.objects.filter(is_active=True).first()
    
    # Obtener todos los modelos para comparación
    all_models = MLModel.objects.all().order_by('-f1_score')
    
    context = {
        'total_objects': total_objects,
        'confirmed_count': confirmed_count,
        'candidate_count': candidate_count,
        'false_positive_count': false_positive_count,
        'confirmed_percentage': round((confirmed_count / total_objects) * 100, 1) if total_objects > 0 else 0,
        'candidate_percentage': round((candidate_count / total_objects) * 100, 1) if total_objects > 0 else 0,
        'false_positive_percentage': round((false_positive_count / total_objects) * 100, 1) if total_objects > 0 else 0,
        'active_model': active_model,
        'all_models': all_models,
    }
    
    return render(request, 'exoplanet_ai/prediction_dashboard.html', context)


@csrf_exempt
def predict_exoplanet(request):
    """
    API endpoint para realizar predicciones de exoplanetas
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Método no permitido'}, status=405)
    
    try:
        # Obtener datos del request
        data = json.loads(request.body)
        
        # Validar que se proporcionen todas las características necesarias
        required_features = [
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
        
        # Verificar características faltantes
        missing_features = [f for f in required_features if f not in data]
        if missing_features:
            return JsonResponse({
                'error': f'Características faltantes: {", ".join(missing_features)}'
            }, status=400)
        
        # Obtener modelo activo
        active_model = MLModel.objects.filter(is_active=True).first()
        if not active_model:
            return JsonResponse({'error': 'No hay modelo activo disponible'}, status=500)
        
        # Cargar el modelo y el preprocessor
        try:
            model = joblib.load(active_model.model_file_path)
            preprocessor = joblib.load('ml_models/preprocessor.pkl')
        except Exception as e:
            return JsonResponse({'error': f'Error cargando modelo: {str(e)}'}, status=500)
        
        # Preparar datos para predicción
        input_data = pd.DataFrame([{
            feature: float(data[feature]) for feature in required_features
        }])
        
        # Aplicar el mismo preprocesamiento
        input_scaled = preprocessor.scaler.transform(input_data)
        
        # Realizar predicción
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Mapear predicción a etiqueta
        label_mapping = {0: 'CONFIRMED', 1: 'CANDIDATE', 2: 'FALSE_POSITIVE'}
        predicted_label = label_mapping[prediction]
        
        # Preparar probabilidades
        probabilities = {
            'CONFIRMED': float(prediction_proba[0]),
            'CANDIDATE': float(prediction_proba[1]), 
            'FALSE_POSITIVE': float(prediction_proba[2])
        }
        
        # Obtener confianza (probabilidad máxima)
        confidence = float(max(prediction_proba))
        
        return JsonResponse({
            'success': True,
            'prediction': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities,
            'model_used': active_model.name,
            'model_accuracy': active_model.accuracy,
            'input_data': data
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'JSON inválido'}, status=400)
    except Exception as e:
        return JsonResponse({'error': f'Error interno: {str(e)}'}, status=500)


def model_comparison(request):
    """
    Vista para comparar diferentes modelos de ML
    """
    models = MLModel.objects.all().order_by('-f1_score')
    
    context = {
        'models': models
    }
    
    return render(request, 'exoplanet_ai/model_comparison.html', context)


@csrf_exempt
def switch_active_model(request):
    """
    API endpoint para cambiar el modelo activo
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Método no permitido'}, status=405)
    
    try:
        data = json.loads(request.body)
        model_id = data.get('model_id')
        
        if not model_id:
            return JsonResponse({'error': 'model_id es requerido'}, status=400)
        
        # Desactivar todos los modelos
        MLModel.objects.all().update(is_active=False)
        
        # Activar el modelo seleccionado
        model = get_object_or_404(MLModel, id=model_id)
        model.is_active = True
        model.save()
        
        return JsonResponse({
            'success': True,
            'active_model': model.name,
            'message': f'Modelo {model.name} activado exitosamente'
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