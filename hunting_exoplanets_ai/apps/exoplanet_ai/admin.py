from django.contrib import admin
from .models import ExoplanetData, MLModel


@admin.register(ExoplanetData)
class ExoplanetDataAdmin(admin.ModelAdmin):
    """
    Configuración del admin para visualizar y gestionar datos de exoplanetas
    """
    list_display = [
        'object_name', 
        'source_mission',
        'original_disposition',
        'ml_prediction',
        'orbital_period',
        'planetary_radius',
        'is_confirmed_exoplanet',
        'created_at'
    ]
    
    list_filter = [
        'source_mission',
        'original_disposition',
        'ml_prediction',
        'created_at',
    ]
    
    search_fields = [
        'object_name',
        'planet_name',
        'object_id',
    ]
    
    readonly_fields = [
        'created_at',
        'updated_at',
        'is_confirmed_exoplanet',
        'ml_accuracy_for_object',
    ]
    
    fieldsets = (
        ('Identificación', {
            'fields': (
                'source_mission',
                'object_id',
                'object_name',
                'planet_name',
            )
        }),
        ('Variables para Machine Learning', {
            'fields': (
                'orbital_period',
                'orbital_period_err',
                'transit_duration', 
                'transit_duration_err',
                'planetary_radius',
                'planetary_radius_err',
            )
        }),
        ('Variables Adicionales', {
            'fields': (
                'transit_epoch',
                'transit_depth',
                'impact_parameter',
                'equilibrium_temperature',
            ),
            'classes': ('collapse',)
        }),
        ('Clasificaciones', {
            'fields': (
                'original_disposition',
                'ml_prediction',
                'ml_confidence',
                'disposition_score',
            )
        }),
        ('Información Estelar', {
            'fields': (
                'stellar_temperature',
                'stellar_radius',
                'stellar_mass',
            ),
            'classes': ('collapse',)
        }),
        ('Metadatos', {
            'fields': (
                'data_source_file',
                'notes',
                'created_at',
                'updated_at',
                'is_confirmed_exoplanet',
                'ml_accuracy_for_object',
            ),
            'classes': ('collapse',)
        }),
    )
    
    # Filtros avanzados
    def get_queryset(self, request):
        return super().get_queryset(request).select_related()
    
    # Acciones personalizadas
    actions = ['mark_as_confirmed', 'clear_ml_predictions']
    
    def mark_as_confirmed(self, request, queryset):
        """Marca los objetos seleccionados como exoplanetas confirmados"""
        updated = queryset.update(original_disposition='CONFIRMED')
        self.message_user(request, f'{updated} objetos marcados como confirmados.')
    mark_as_confirmed.short_description = "Marcar como exoplanetas confirmados"
    
    def clear_ml_predictions(self, request, queryset):
        """Limpia las predicciones ML de los objetos seleccionados"""
        updated = queryset.update(ml_prediction=None, ml_confidence=None)
        self.message_user(request, f'Predicciones ML eliminadas de {updated} objetos.')
    clear_ml_predictions.short_description = "Limpiar predicciones ML"


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    """
    Configuración del admin para gestionar modelos de ML
    """
    list_display = [
        'name',
        'algorithm',
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'is_active',
        'created_at'
    ]
    
    list_filter = [
        'algorithm',
        'is_active',
        'created_at',
    ]
    
    search_fields = [
        'name',
        'algorithm',
    ]
    
    readonly_fields = [
        'created_at',
    ]
    
    fieldsets = (
        ('Información del Modelo', {
            'fields': (
                'name',
                'algorithm',
                'is_active',
            )
        }),
        ('Métricas de Rendimiento', {
            'fields': (
                'accuracy',
                'precision',
                'recall',
                'f1_score',
            )
        }),
        ('Datos de Entrenamiento', {
            'fields': (
                'training_size',
                'test_size',
                'features_used',
                'hyperparameters',
            )
        }),
        ('Archivos del Modelo', {
            'fields': (
                'model_file_path',
                'scaler_file_path',
            )
        }),
        ('Metadatos', {
            'fields': (
                'notes',
                'created_at',
            )
        }),
    )
    
    # Solo permitir un modelo activo a la vez
    def save_model(self, request, obj, form, change):
        if obj.is_active:
            # Desactivar otros modelos
            MLModel.objects.filter(is_active=True).update(is_active=False)
        super().save_model(request, obj, form, change)
    
    actions = ['activate_model', 'deactivate_model']
    
    def activate_model(self, request, queryset):
        """Activa el modelo seleccionado (desactiva los demás)"""
        if queryset.count() != 1:
            self.message_user(request, 'Por favor selecciona exactamente un modelo.', level='error')
            return
        
        # Desactivar todos los modelos
        MLModel.objects.update(is_active=False)
        # Activar el seleccionado
        model = queryset.first()
        model.is_active = True
        model.save()
        
        self.message_user(request, f'Modelo "{model.name}" activado exitosamente.')
    activate_model.short_description = "Activar modelo seleccionado"
    
    def deactivate_model(self, request, queryset):
        """Desactiva los modelos seleccionados"""
        updated = queryset.update(is_active=False)
        self.message_user(request, f'{updated} modelos desactivados.')
    deactivate_model.short_description = "Desactivar modelos seleccionados"
