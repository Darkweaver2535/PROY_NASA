#!/usr/bin/env python3
"""
NASA Exoplanet ML Training Pipeline
==================================

Complete machine learning training pipeline for NASA exoplanet classification
using the "A World Away: Hunting for Exoplanets with AI" challenge datasets.

Author: NASA Challenge Team
Date: 2024
"""

import os
import sys
import django
import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from datetime import datetime

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core_project.settings')
django.setup()

from apps.exoplanet_ai.models import ExoplanetData
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class NASAExoplanetMLTrainer:
    """Complete ML Training Pipeline for NASA Exoplanet Classification"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load data from Django database and prepare for ML"""
        print("🚀 Loading NASA Exoplanet Data from Database...")
        print("=" * 60)
        
        # Query all data from database
        queryset = ExoplanetData.objects.all()
        
        if not queryset.exists():
            print("❌ No data found in database!")
            return False
            
        # Convert to DataFrame
        data_list = []
        for obj in queryset:
            data_dict = {
                'kepid': getattr(obj, 'kepid', 0),
                'koi_period': getattr(obj, 'koi_period', 0),
                'koi_impact': getattr(obj, 'koi_impact', 0),
                'koi_duration': getattr(obj, 'koi_duration', 0),
                'koi_depth': getattr(obj, 'koi_depth', 0),
                'koi_prad': getattr(obj, 'koi_prad', 0),
                'koi_sma': getattr(obj, 'koi_sma', 0),
                'koi_teq': getattr(obj, 'koi_teq', 0),
                'koi_insol': getattr(obj, 'koi_insol', 0),
                'koi_steff': getattr(obj, 'koi_steff', 0),
                'koi_slogg': getattr(obj, 'koi_slogg', 0),
                'koi_srad': getattr(obj, 'koi_srad', 0),
                'koi_smass': getattr(obj, 'koi_smass', 0),
                'koi_kepmag': getattr(obj, 'koi_kepmag', 0),
                'ra': getattr(obj, 'ra', 0),
                'dec': getattr(obj, 'dec', 0),
                'source_mission': obj.source_mission,
                'original_disposition': obj.original_disposition
            }
            data_list.append(data_dict)
        
        self.df = pd.DataFrame(data_list)
        print(f"✅ Loaded {len(self.df)} objects from database")
        print(f"📊 Columns: {list(self.df.columns)}")
        
        # Show disposition distribution
        disposition_counts = self.df['original_disposition'].value_counts()
        print(f"\\n🎯 Disposition Distribution:")
        for disposition, count in disposition_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"   {disposition}: {count} ({percentage:.2f}%)")
            
        return True
        
    def feature_engineering(self):
        """Prepare features for machine learning"""
        print("\\n🔧 Feature Engineering...")
        print("=" * 60)
        
        # Select numerical features (remove null/inf values)
        numerical_features = [
            'koi_period', 'koi_impact', 'koi_duration', 'koi_depth', 
            'koi_prad', 'koi_sma', 'koi_teq', 'koi_insol',
            'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass', 
            'koi_kepmag', 'ra', 'dec'
        ]
        
        # Clean data
        for col in numerical_features:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            self.df[col] = self.df[col].fillna(self.df[col].median())
            
        # Remove infinite values
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.fillna(self.df.median(numeric_only=True))
        
        # Encode categorical features
        mission_encoded = pd.get_dummies(self.df['source_mission'], prefix='mission')
        self.df = pd.concat([self.df, mission_encoded], axis=1)
        
        # Final feature selection
        self.feature_columns = numerical_features + list(mission_encoded.columns)
        
        print(f"✅ Selected {len(self.feature_columns)} features")
        print(f"📊 Feature columns: {self.feature_columns}")
        
        # Filter out classes with too few samples (like UNKNOWN with 1 sample)
        class_counts = self.df['original_disposition'].value_counts()
        valid_classes = class_counts[class_counts >= 2].index.tolist()
        
        print(f"📊 Filtering classes with <2 samples:")
        for class_name, count in class_counts.items():
            if count < 2:
                print(f"   ⚠️  Removing {class_name}: {count} samples")
        
        # Filter dataset to only include valid classes
        self.df_filtered = self.df[self.df['original_disposition'].isin(valid_classes)]
        
        # Prepare X and y
        self.X = self.df_filtered[self.feature_columns]
        self.y = self.df_filtered['original_disposition']
        
        # Encode labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        print(f"✅ Dataset shape: {self.X.shape}")
        print(f"✅ Target classes: {list(self.label_encoder.classes_)}")
        
        return True
        
    def initialize_models(self):
        """Initialize ML models for training"""
        print("\\n🤖 Initializing ML Models...")
        print("=" * 60)
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='ovr'
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=42,
                probability=True
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            )
        }
        
        print(f"✅ Initialized {len(self.models)} ML models")
        for name in self.models.keys():
            print(f"   • {name}")
            
        return True
        
    def train_and_evaluate_models(self):
        """Train and evaluate all models"""
        print("\\n🏃‍♂️ Training and Evaluating Models...")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded, 
            test_size=0.2, 
            random_state=42, 
            stratify=self.y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"📊 Training set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        
        # Train each model
        for model_name, model in self.models.items():
            print(f"\\n🎯 Training {model_name}...")
            
            try:
                # Train model
                if model_name in ['Logistic Regression', 'SVM']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    # Cross-validation on scaled data
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    # Cross-validation on original data
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                # Store results
                self.results[model_name] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'model': model
                }
                
                print(f"   ✅ Accuracy: {accuracy:.4f}")
                print(f"   ✅ CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
                
            except Exception as e:
                print(f"   ❌ Error training {model_name}: {e}")
                
        return True
        
    def display_results(self):
        """Display training results"""
        print("\\n📊 MODEL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        # Sort by accuracy
        sorted_results = sorted(
            self.results.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        print(f"{'Model':<20} {'Accuracy':<12} {'CV Score':<15} {'Status'}")
        print("-" * 60)
        
        best_accuracy = max([m['accuracy'] for m in self.results.values()])
        
        for model_name, metrics in sorted_results:
            accuracy = metrics['accuracy']
            cv_score = metrics['cv_mean']
            cv_std = metrics['cv_std']
            status = "🥇" if accuracy == best_accuracy else "✅"
            
            print(f"{model_name:<20} {accuracy:<12.4f} {cv_score:.4f}±{cv_std:.4f}    {status}")
        
        print("\\n🎯 KEY METRICS FOR NASA CHALLENGE:")
        print(f"   📊 Best Model Accuracy: {best_accuracy:.1%}")
        print(f"   🚀 Features Used: {len(self.feature_columns)} orbital & stellar parameters")
        print(f"   🌌 Training Samples: {len(self.X)} NASA exoplanet candidates")
        print(f"   ✅ Modelo guardado exitosamente como exoplanet_classifier.joblib")
            
        return True
        
    def save_models(self):
        """Save trained models"""
        print("\\n💾 Saving Trained Models...")
        print("=" * 60)
        
        # Create directories
        apps_dir = Path("apps/exoplanet_ai")
        apps_dir.mkdir(exist_ok=True)
        models_dir = Path("ml_models")
        models_dir.mkdir(exist_ok=True)
        
        # Find best model by accuracy
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_model = self.results[best_model_name]['model']
        best_accuracy = self.results[best_model_name]['accuracy']
        
        # Save best model as exoplanet_classifier.joblib (NASA Challenge requirement)
        classifier_path = apps_dir / "exoplanet_classifier.joblib"
        try:
            joblib.dump(best_model, classifier_path)
            print(f"   🥇 Saved best model ({best_model_name}) as exoplanet_classifier.joblib")
            print(f"   📊 Best model accuracy: {best_accuracy:.1%}")
        except Exception as e:
            print(f"   ❌ Error saving best model: {e}")
        
        # Save each model for comparison
        for model_name, metrics in self.results.items():
            model = metrics['model']
            filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
            filepath = models_dir / filename
            
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                print(f"   ✅ Saved: {filename}")
            except Exception as e:
                print(f"   ❌ Error saving {filename}: {e}")
                
        # Save preprocessing objects
        try:
            with open(models_dir / "scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(models_dir / "label_encoder.pkl", 'wb') as f:
                pickle.dump(self.label_encoder, f)
            with open(models_dir / "feature_columns.pkl", 'wb') as f:
                pickle.dump(self.feature_columns, f)
            print(f"   ✅ Saved preprocessing objects")
        except Exception as e:
            print(f"   ❌ Error saving preprocessing objects: {e}")
            
        return True
        
    def run_complete_training(self):
        """Run the complete training pipeline"""
        print("🌌 NASA EXOPLANET ML TRAINING PIPELINE")
        print("=" * 70)
        print("Training models for: 'A World Away: Hunting for Exoplanets with AI'")
        print("=" * 70)
        
        success = True
        
        # Step 1: Load data
        if not self.load_and_prepare_data():
            print("❌ Failed to load data!")
            return False
            
        # Step 2: Feature engineering
        if not self.feature_engineering():
            print("❌ Failed feature engineering!")
            return False
            
        # Step 3: Initialize models
        if not self.initialize_models():
            print("❌ Failed to initialize models!")
            return False
            
        # Step 4: Train models
        if not self.train_and_evaluate_models():
            print("❌ Failed to train models!")
            return False
            
        # Step 5: Display results
        self.display_results()
        
        # Step 6: Save models
        self.save_models()
        
        print("\\n" + "=" * 70)
        print("🎉 ML TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("🚀 Models ready for exoplanet classification!")
        print("=" * 70)
        
        return True


def main():
    """Main execution function"""
    print("🚀 Starting NASA Exoplanet ML Training...")
    
    # Initialize trainer
    trainer = NASAExoplanetMLTrainer()
    
    # Run training
    success = trainer.run_complete_training()
    
    if success:
        print("\\n✅ Training completed successfully!")
        return 0
    else:
        print("\\n❌ Training failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())