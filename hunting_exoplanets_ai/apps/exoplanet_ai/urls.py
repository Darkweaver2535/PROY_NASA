from django.urls import path
from . import views
from . import prediction_views
from . import data_views
from . import training_views

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
    
    # URLs para gestión de datos
    path('data/', data_views.data_management_dashboard, name='data_management'),
    path('data/upload/', data_views.upload_csv_data, name='upload_csv'),
    path('data/preview/', data_views.validate_data_preview, name='preview_csv'),
    path('data/batch-predict/', data_views.batch_predict_csv, name='batch_predict'),
    path('data/download-sample/', data_views.download_sample_csv, name='download_sample'),
    
    # URLs para entrenamiento de modelos
    path('training/', training_views.model_training_dashboard, name='model_training'),
    path('training/start/', training_views.start_model_training, name='start_training'),
    path('training/status/', training_views.get_training_status, name='training_status'),
    path('training/metrics/', training_views.get_training_metrics, name='training_metrics'),
    path('training/hyperparameters/', training_views.hyperparameter_tuning, name='hyperparameter_tuning'),
]