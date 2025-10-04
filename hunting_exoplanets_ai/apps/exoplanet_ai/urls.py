from django.urls import path
from . import views

app_name = 'exoplanet_ai'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('api/data/', views.api_exoplanet_data, name='api_data'),
    path('models/', views.ml_models_view, name='ml_models'),
    path('exoplanet/<int:exoplanet_id>/', views.exoplanet_detail, name='exoplanet_detail'),
]