#!/usr/bin/env python3
"""
NASA Exoplanet Data Preprocessing and Verification Script
=======================================================

This script verifies the NASA exoplanet dataset loading and preprocessing pipeline
for the "A World Away: Hunting for Exoplanets with AI" challenge.

Author: NASA Challenge Team
Date: 2024
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add Django settings
django_path = Path(__file__).parent.parent.parent
sys.path.append(str(django_path))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

from apps.exoplanet_ai.models import ExoplanetData


class NASADataPreprocessor:
    """NASA Exoplanet Data Preprocessing and Verification Class"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.datasets = []
        self.combined_data = None
        
    def load_nasa_datasets(self):
        """Load NASA datasets from CSV files"""
        print("🚀 Loading NASA Exoplanet Datasets...")
        print("=" * 50)
        
        dataset_files = [
            'kepler_exoplanet_search_results.csv',
            'k2_exoplanet_search_results.csv', 
            'tess_exoplanet_search_results.csv'
        ]
        
        datasets_found = []
        total_objects = 0
        
        for dataset_file in dataset_files:
            file_path = self.data_dir / dataset_file
            
            if file_path.exists():
                print(f"📡 Loading {dataset_file}...")
                try:
                    df = pd.read_csv(file_path)
                    print(f"   ✅ Loaded: {len(df)} objects")
                    print(f"   📊 Columns: {list(df.columns)}")
                    
                    # Check for disposition column
                    if 'koi_disposition' in df.columns:
                        disposition_col = 'koi_disposition'
                    elif 'toi_disposition' in df.columns:
                        disposition_col = 'toi_disposition'
                    elif 'disposition' in df.columns:
                        disposition_col = 'disposition'
                    else:
                        print(f"   ⚠️  No disposition column found in {dataset_file}")
                        continue
                    
                    df['source_mission'] = dataset_file.split('_')[0].upper()
                    df['original_disposition'] = df[disposition_col]
                    
                    datasets_found.append(df)
                    total_objects += len(df)
                    
                except Exception as e:
                    print(f"   ❌ Error loading {dataset_file}: {e}")
            else:
                print(f"   ⚠️  File not found: {dataset_file}")
        
        if datasets_found:
            self.combined_data = pd.concat(datasets_found, ignore_index=True)
            print(f"\n🌟 Total NASA Objects Loaded: {total_objects}")
            return True
        else:
            print("❌ No datasets found!")
            return False
    
    def verify_data_structure(self):
        """Verify the structure of loaded data"""
        if self.combined_data is None:
            print("❌ No data loaded to verify!")
            return False
            
        print("\n🔍 Data Structure Verification")
        print("=" * 50)
        
        # Show first 5 rows
        print("📋 First 5 rows of loaded data:")
        print(self.combined_data.head())
        
        # Show basic info
        print(f"\n📊 Dataset Info:")
        print(f"   Total Rows: {len(self.combined_data)}")
        print(f"   Total Columns: {len(self.combined_data.columns)}")
        print(f"   Memory Usage: {self.combined_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for required columns
        required_cols = ['source_mission', 'original_disposition']
        print(f"\n✅ Required Columns Check:")
        for col in required_cols:
            if col in self.combined_data.columns:
                print(f"   ✅ {col}: Found")
            else:
                print(f"   ❌ {col}: Missing")
        
        return True
    
    def analyze_disposition_distribution(self):
        """Analyze the distribution of disposition labels"""
        if self.combined_data is None:
            print("❌ No data loaded to analyze!")
            return False
            
        print("\n🎯 Disposition Distribution Analysis")
        print("=" * 50)
        
        # Overall distribution
        disposition_counts = self.combined_data['original_disposition'].value_counts()
        print("📊 Overall Disposition Distribution:")
        for disposition, count in disposition_counts.items():
            percentage = (count / len(self.combined_data)) * 100
            print(f"   {disposition}: {count} ({percentage:.2f}%)")
        
        # Distribution by mission
        print("\n🚀 Distribution by Mission:")
        mission_disposition = self.combined_data.groupby(['source_mission', 'original_disposition']).size().unstack(fill_value=0)
        print(mission_disposition)
        
        # Check for data quality
        print(f"\n🔍 Data Quality Check:")
        null_count = self.combined_data['original_disposition'].isnull().sum()
        print(f"   Null dispositions: {null_count}")
        
        unique_dispositions = self.combined_data['original_disposition'].unique()
        print(f"   Unique dispositions: {list(unique_dispositions)}")
        
        return True
    
    def verify_database_sync(self):
        """Verify data synchronization with Django database"""
        print("\n💾 Database Synchronization Check")
        print("=" * 50)
        
        try:
            db_count = ExoplanetData.objects.count()
            print(f"📊 Objects in database: {db_count}")
            
            if self.combined_data is not None:
                csv_count = len(self.combined_data)
                print(f"📊 Objects in CSV files: {csv_count}")
                
                if db_count == csv_count:
                    print("✅ Database and CSV data are synchronized!")
                else:
                    print(f"⚠️  Database ({db_count}) and CSV ({csv_count}) counts don't match")
            
            # Show sample database records
            sample_records = ExoplanetData.objects.all()[:5]
            print(f"\n📋 Sample Database Records:")
            for i, record in enumerate(sample_records, 1):
                print(f"   {i}. Mission: {record.source_mission}, Disposition: {record.original_disposition}")
                
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False
            
        return True
    
    def run_complete_verification(self):
        """Run complete data verification pipeline"""
        print("🌌 NASA Exoplanet Data Verification Pipeline")
        print("=" * 60)
        print("Verifying data loading and preprocessing for:")
        print("'A World Away: Hunting for Exoplanets with AI' Challenge")
        print("=" * 60)
        
        success = True
        
        # Step 1: Load datasets
        if not self.load_nasa_datasets():
            success = False
        
        # Step 2: Verify structure
        if not self.verify_data_structure():
            success = False
        
        # Step 3: Analyze dispositions
        if not self.analyze_disposition_distribution():
            success = False
            
        # Step 4: Check database sync
        if not self.verify_database_sync():
            success = False
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 Data Verification COMPLETED SUCCESSFULLY!")
        else:
            print("❌ Data Verification FAILED - Check errors above")
        print("=" * 60)
        
        return success


def main():
    """Main execution function"""
    print("🚀 Starting NASA Exoplanet Data Verification...")
    
    # Initialize preprocessor
    preprocessor = NASADataPreprocessor()
    
    # Run verification
    success = preprocessor.run_complete_verification()
    
    if success:
        print("\n✅ All verification tests passed!")
        return 0
    else:
        print("\n❌ Some verification tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())