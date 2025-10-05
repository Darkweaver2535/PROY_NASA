import json
import uuid
import threading
import time
from datetime import datetime
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import ExoplanetData

class ModelTrainingStatus:
    def __init__(self, training_id, algorithm, status='pending', progress=0):
        self.training_id = training_id
        self.algorithm = algorithm
        self.status = status
        self.progress = progress
        self.started_at = datetime.now()
        self.completed_at = None
        self.accuracy = None

training_sessions = {}

def model_training_dashboard(request):
    context = {
        'total_objects': ExoplanetData.objects.count(),
        'confirmed_planets': ExoplanetData.objects.filter(original_disposition='CONFIRMED').count(),
        'false_positives': ExoplanetData.objects.filter(original_disposition='FALSE_POSITIVE').count(),
        'candidates': ExoplanetData.objects.filter(original_disposition='CANDIDATE').count(),
        'available_algorithms': [
            'random_forest',
            'gradient_boosting', 
            'svm',
            'logistic_regression',
            'knn'
        ]
    }
    return render(request, 'exoplanet_ai/model_training.html', context)

@csrf_exempt
def start_model_training(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Método no permitido'}, status=405)
    
    try:
        data = json.loads(request.body)
        algorithm = data.get('algorithm', 'random_forest')
        
        training_id = uuid.uuid4()
        training_status = ModelTrainingStatus(training_id, algorithm, 'starting', 0)
        training_sessions[str(training_id)] = training_status
        
        thread = threading.Thread(target=run_model_training, args=(str(training_id), algorithm))
        thread.daemon = True
        thread.start()
        
        return JsonResponse({
            'status': 'success',
            'training_id': str(training_id),
            'message': f'Entrenamiento de {algorithm} iniciado'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def get_training_status(request):
    training_id = request.GET.get('training_id')
    if not training_id or training_id not in training_sessions:
        return JsonResponse({'error': 'ID inválido'}, status=404)
    
    status = training_sessions[training_id]
    return JsonResponse({
        'training_id': training_id,
        'algorithm': status.algorithm,
        'status': status.status,
        'progress': status.progress,
        'accuracy': status.accuracy
    })

@csrf_exempt
def get_training_metrics(request):
    return JsonResponse({'metrics': 'placeholder'})

@csrf_exempt
def hyperparameter_tuning(request):
    return JsonResponse({'configs': 'placeholder'})

def run_model_training(training_id, algorithm):
    try:
        status = training_sessions[training_id]
        status.status = 'running'
        status.progress = 10
        
        time.sleep(2)
        status.progress = 50
        
        time.sleep(3)
        status.progress = 100
        status.status = 'completed'
        status.accuracy = 0.80
        status.completed_at = datetime.now()
        
    except Exception as e:
        status.status = 'error'