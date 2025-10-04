from django.shortcuts import render
from django.http import JsonResponse
from django.db.models import Count, Q, Min, Max, Avg
from .models import ExoplanetData, MLModel


def dashboard_view(request):
    """
    Vista principal del dashboard con estadísticas de exoplanetas
    """
    # Estadísticas generales
    total_objects = ExoplanetData.objects.count()
    confirmed_exoplanets = ExoplanetData.objects.filter(original_disposition='CONFIRMED').count()
    candidates = ExoplanetData.objects.filter(original_disposition='CANDIDATE').count()
    false_positives = ExoplanetData.objects.filter(original_disposition='FALSE_POSITIVE').count()
    
    # Estadísticas por misión con porcentajes
    mission_stats = ExoplanetData.objects.values('source_mission').annotate(
        count=Count('id')
    ).order_by('-count')
    
    # Calcular porcentajes para misiones
    for mission in mission_stats:
        if total_objects > 0:
            mission['percentage'] = round((mission['count'] * 100.0) / total_objects, 1)
        else:
            mission['percentage'] = 0
    
    # Estadísticas por disposición con porcentajes
    disposition_stats = ExoplanetData.objects.values('original_disposition').annotate(
        count=Count('id')
    ).order_by('-count')
    
    # Calcular porcentajes para disposiciones
    for disposition in disposition_stats:
        if total_objects > 0:
            disposition['percentage'] = round((disposition['count'] * 100.0) / total_objects, 1)
        else:
            disposition['percentage'] = 0
    
    # Rangos de características para ML
    orbital_period_range = ExoplanetData.objects.filter(
        orbital_period__isnull=False
    ).aggregate(
        min_period=Min('orbital_period'),
        max_period=Max('orbital_period'),
        avg_period=Avg('orbital_period')
    )
    
    planetary_radius_range = ExoplanetData.objects.filter(
        planetary_radius__isnull=False
    ).aggregate(
        min_radius=Min('planetary_radius'),
        max_radius=Max('planetary_radius'),
        avg_radius=Avg('planetary_radius')
    )
    
    context = {
        'total_objects': total_objects,
        'confirmed_exoplanets': confirmed_exoplanets,
        'candidates': candidates,
        'false_positives': false_positives,
        'mission_stats': mission_stats,
        'disposition_stats': disposition_stats,
        'orbital_period_range': orbital_period_range,
        'planetary_radius_range': planetary_radius_range,
    }
    
    return render(request, 'exoplanet_ai/dashboard.html', context)


def api_exoplanet_data(request):
    """
    API endpoint para obtener datos de exoplanetas en formato JSON
    """
    # Filtros opcionales
    mission = request.GET.get('mission')
    disposition = request.GET.get('disposition')
    limit = int(request.GET.get('limit', 100))
    
    queryset = ExoplanetData.objects.all()
    
    if mission:
        queryset = queryset.filter(source_mission=mission)
    
    if disposition:
        queryset = queryset.filter(original_disposition=disposition)
    
    # Seleccionar campos relevantes
    data = list(queryset.values(
        'id',
        'object_name',
        'source_mission',
        'original_disposition',
        'orbital_period',
        'transit_duration',
        'planetary_radius',
        'ml_prediction',
        'ml_confidence'
    )[:limit])
    
    return JsonResponse({
        'count': len(data),
        'data': data
    })


def ml_models_view(request):
    """
    Vista para mostrar información sobre los modelos de ML
    """
    active_model = MLModel.objects.filter(is_active=True).first()
    all_models = MLModel.objects.all().order_by('-created_at')
    
    context = {
        'active_model': active_model,
        'all_models': all_models,
    }
    
    return render(request, 'exoplanet_ai/ml_models.html', context)


def exoplanet_detail(request, exoplanet_id):
    """
    Vista de detalle para un exoplaneta específico
    """
    try:
        exoplanet = ExoplanetData.objects.get(id=exoplanet_id)
        
        # Obtener características para ML
        ml_features = exoplanet.get_features_for_ml()
        
        context = {
            'exoplanet': exoplanet,
            'ml_features': ml_features,
        }
        
        return render(request, 'exoplanet_ai/exoplanet_detail.html', context)
        
    except ExoplanetData.DoesNotExist:
        return render(request, 'exoplanet_ai/404.html', status=404)
