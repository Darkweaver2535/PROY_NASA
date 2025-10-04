from django.urls import path
from . import views
from . import prediction_views

app_name = 'exoplanet_ai'

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('api/data/', views.api_exoplanet_data, name='api_data'),
    path('models/', views.ml_models_view, name='ml_models'),
    path('exoplanet/<int:exoplanet_id>/', views.exoplanet_detail, name='exoplanet_detail'),
    
    # URLs para predicciones ML
    path('predict/', prediction_views.prediction_dashboard, name='prediction_dashboard'),
    path('api/predict/', prediction_views.predict_exoplanet, name='api_predict'),
    path('api/sample-data/', prediction_views.get_sample_data, name='api_sample_data'),
    path('models/comparison/', prediction_views.model_comparison, name='model_comparison'),
    path('api/switch-model/', prediction_views.switch_active_model, name='api_switch_model'),
]