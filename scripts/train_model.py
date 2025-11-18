"""
Master training pipeline
Runs: Data download → Preprocessing → Feature extraction → Training → Evaluation
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (
    load_config,
    download_dataset_from_kaggle,
    save_features,
    load_features,
    save_model
)
from src.preprocessing import load_and_preprocess_dataset
from src.features import extract_features_from_images
from src.train import train_model
from src.evaluate import evaluate_model
import numpy as np

def main():
    """Run complete training pipeline"""
    
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # ============================================
    # STEP 1: DOWNLOAD DATASET (if needed)
    # ============================================
    print("\n" + "="*60)
    print("STEP 1: DOWNLOAD DATASET")
    print("="*60)
    
    if not os.path.exists(config['data']['raw_dir']):
        download_dataset_from_kaggle(
            dataset_name=config['data']['kaggle_dataset'],
            output_dir=config['data']['raw_dir']
        )
    else:
        print(f"Dataset already exists at {config['data']['raw_dir']}")
    
    # ============================================
    # STEP 2: PREPROCESS IMAGES
    # ============================================
    print("\n" + "="*60)
    print("STEP 2: PREPROCESSING IMAGES")
    print("="*60)
    
    print("\nProcessing training data...")
    train_images, y_train, _ = load_and_preprocess_dataset(
        csv_path=config['data']['train_csv'],
        image_folder=config['data']['raw_dir']
    )
    
    print("\nProcessing test data...")
    test_images, y_test, _ = load_and_preprocess_dataset(
        csv_path=config['data']['test_csv'],
        image_folder=config['data']['raw_dir']
    )
    
    # ============================================
    # STEP 3: EXTRACT FEATURES
    # ============================================
    print("\n" + "="*60)
    print("STEP 3: EXTRACTING FEATURES")
    print("="*60)
    
    print("\nExtracting training features...")
    X_train = extract_features_from_images(train_images)
    
    print("\nExtracting test features...")
    X_test = extract_features_from_images(test_images)
    
    # Save features
    save_features(X_train, y_train, config['data']['train_features'])
    save_features(X_test, y_test, config['data']['test_features'])
    
    # ============================================
    # STEP 4: TRAIN MODEL
    # ============================================
    model, scaler = train_model(X_train, y_train, config['model'])
    
    # ============================================
    # STEP 5: EVALUATE MODEL
    # ============================================
    results = evaluate_model(model, scaler, X_test, y_test)
    
    # ============================================
    # STEP 6: SAVE MODEL
    # ============================================
    save_model(model, scaler, config['model']['save_path'])
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"Final Test Accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()