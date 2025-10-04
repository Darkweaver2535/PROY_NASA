from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator


class ExoplanetData(models.Model):
    """
    Modelo para almacenar datos de exoplanetas de las misiones NASA:
    Kepler Objects of Interest, K2 Planets and Candidates, y TESS Project Candidates
    """
    
    # Identificación y metadatos
    source_mission = models.CharField(
        max_length=20,
        choices=[
            ('KEPLER', 'Kepler Objects of Interest'),
            ('K2', 'K2 Planets and Candidates'),
            ('TESS', 'TESS Project Candidates'),
        ],
        help_text="Misión NASA de origen de los datos"
    )
    
    object_id = models.CharField(
        max_length=50,
        help_text="ID único del objeto (kepid, epic_id, tic_id, etc.)"
    )
    
    object_name = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Nombre del objeto (KOI, EPIC, TOI, etc.)"
    )
    
    planet_name = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Nombre del planeta confirmado (ej: Kepler-452b)"
    )
    
    # Variables principales para clasificación ML (Features)
    orbital_period = models.FloatField(
        validators=[MinValueValidator(0.0)],
        help_text="Período orbital en días"
    )
    
    orbital_period_err = models.FloatField(
        blank=True,
        null=True,
        help_text="Error del período orbital"
    )
    
    transit_duration = models.FloatField(
        validators=[MinValueValidator(0.0)],
        help_text="Duración del tránsito en horas"
    )
    
    transit_duration_err = models.FloatField(
        blank=True,
        null=True,
        help_text="Error de la duración del tránsito"
    )
    
    planetary_radius = models.FloatField(
        validators=[MinValueValidator(0.0)],
        help_text="Radio planetario en radios terrestres"
    )
    
    planetary_radius_err = models.FloatField(
        blank=True,
        null=True,
        help_text="Error del radio planetario"
    )
    
    # Variables adicionales relevantes para ML
    transit_epoch = models.FloatField(
        blank=True,
        null=True,
        help_text="Época de tránsito (BJD - 2454833)"
    )
    
    transit_depth = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0)],
        help_text="Profundidad del tránsito en ppm"
    )
    
    impact_parameter = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Parámetro de impacto (0-1)"
    )
    
    equilibrium_temperature = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0)],
        help_text="Temperatura de equilibrio en Kelvin"
    )
    
    # Clasificaciones (Target variable)
    DISPOSITION_CHOICES = [
        ('CONFIRMED', 'Confirmed Exoplanet'),
        ('CANDIDATE', 'Planet Candidate'), 
        ('FALSE_POSITIVE', 'False Positive'),
        ('NOT_TRANSIT', 'Not Transit-Like'),
        ('STELLAR_ECLIPSE', 'Stellar Eclipse'),
        ('CENTROID_OFFSET', 'Centroid Offset'),
        ('EPHEMERIS_MATCH', 'Ephemeris Match'),
        ('UNKNOWN', 'Unknown/Unclassified'),
    ]
    
    original_disposition = models.CharField(
        max_length=20,
        choices=DISPOSITION_CHOICES,
        help_text="Clasificación original de astrónomos NASA (ground truth)"
    )
    
    ml_prediction = models.CharField(
        max_length=20,
        choices=DISPOSITION_CHOICES,
        blank=True,
        null=True,
        help_text="Predicción del modelo de Machine Learning"
    )
    
    ml_confidence = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Confianza de la predicción ML (0.0-1.0)"
    )
    
    # Metadatos del modelo
    disposition_score = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Puntuación de disposición original"
    )
    
    # Información estelar del host
    stellar_temperature = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0)],
        help_text="Temperatura estelar efectiva en Kelvin"
    )
    
    stellar_radius = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0)],
        help_text="Radio estelar en radios solares"
    )
    
    stellar_mass = models.FloatField(
        blank=True,
        null=True,
        validators=[MinValueValidator(0.0)],
        help_text="Masa estelar en masas solares"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Campos adicionales para tracking
    data_source_file = models.CharField(
        max_length=255,
        blank=True,
        null=True,
        help_text="Archivo fuente de los datos"
    )
    
    notes = models.TextField(
        blank=True,
        null=True,
        help_text="Notas adicionales sobre el objeto"
    )
    
    class Meta:
        verbose_name = "Exoplanet Data"
        verbose_name_plural = "Exoplanet Data"
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['source_mission']),
            models.Index(fields=['original_disposition']),
            models.Index(fields=['ml_prediction']),
            models.Index(fields=['object_id']),
        ]
        unique_together = ['source_mission', 'object_id']
    
    def __str__(self):
        name = self.planet_name or self.object_name or f"{self.source_mission}-{self.object_id}"
        return f"{name} ({self.original_disposition})"
    
    @property
    def is_confirmed_exoplanet(self):
        """Retorna True si es un exoplaneta confirmado"""
        return self.original_disposition == 'CONFIRMED'
    
    @property
    def ml_accuracy_for_object(self):
        """Calcula si la predicción ML coincide con la disposición original"""
        if self.ml_prediction and self.original_disposition:
            return self.ml_prediction == self.original_disposition
        return None
    
    def get_features_for_ml(self):
        """Retorna un diccionario con las características para ML"""
        return {
            'orbital_period': self.orbital_period,
            'transit_duration': self.transit_duration,
            'planetary_radius': self.planetary_radius,
            'transit_depth': self.transit_depth,
            'impact_parameter': self.impact_parameter,
            'equilibrium_temperature': self.equilibrium_temperature,
            'stellar_temperature': self.stellar_temperature,
            'stellar_radius': self.stellar_radius,
            'stellar_mass': self.stellar_mass,
        }


class MLModel(models.Model):
    """
    Modelo para almacenar información sobre los modelos de ML entrenados
    """
    name = models.CharField(max_length=100, unique=True)
    algorithm = models.CharField(
        max_length=50,
        choices=[
            ('RANDOM_FOREST', 'Random Forest'),
            ('SVM', 'Support Vector Machine'),
            ('NEURAL_NETWORK', 'Neural Network'),
            ('GRADIENT_BOOSTING', 'Gradient Boosting'),
            ('LOGISTIC_REGRESSION', 'Logistic Regression'),
        ]
    )
    
    # Métricas del modelo
    accuracy = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    precision = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    recall = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    f1_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Información de entrenamiento
    training_size = models.IntegerField(validators=[MinValueValidator(1)])
    test_size = models.IntegerField(validators=[MinValueValidator(1)])
    
    features_used = models.JSONField(
        help_text="Lista de características utilizadas para el entrenamiento"
    )
    
    hyperparameters = models.JSONField(
        blank=True,
        null=True,
        help_text="Hiperparámetros del modelo"
    )
    
    # Archivos del modelo
    model_file_path = models.CharField(
        max_length=500,
        help_text="Ruta al archivo del modelo serializado"
    )
    
    scaler_file_path = models.CharField(
        max_length=500,
        blank=True,
        null=True,
        help_text="Ruta al archivo del scaler utilizado"
    )
    
    # Metadatos
    is_active = models.BooleanField(
        default=False,
        help_text="Indica si este es el modelo activo para predicciones"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True, null=True)
    
    class Meta:
        verbose_name = "ML Model"
        verbose_name_plural = "ML Models"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.algorithm}) - Accuracy: {self.accuracy:.3f}"
