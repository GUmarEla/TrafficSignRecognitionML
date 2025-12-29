"""
Main training script for GTSRB traffic sign recognition
Run this script to train the model from scratch
"""
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_and_process_dataset, save_processed_data
from src.train import TrafficSignClassifier


def main():
    """Main training pipeline"""
    
    # ========================================
    # CONFIGURATION
    # ========================================
    # Update these paths for your dataset
    TRAIN_CSV = '/content/gtsrb-german-traffic-sign/Train.csv'
    TEST_CSV = '/content/gtsrb-german-traffic-sign/Test.csv'
    IMAGE_FOLDER = '/content/gtsrb-german-traffic-sign'
    
    # Output paths
    OUTPUT_DIR = '/content/models'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    MODEL_PATH = os.path.join(OUTPUT_DIR, 'rf_model.pkl')
    SCALER_PATH = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    
    # Optional: save processed features for faster re-training
    SAVE_PROCESSED = True
    TRAIN_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'train_features.npz')
    TEST_FEATURES_PATH = os.path.join(OUTPUT_DIR, 'test_features.npz')
    
    # ========================================
    # LOAD AND PROCESS DATA
    # ========================================
    print("="*60)
    print("LOADING AND PROCESSING DATASET")
    print("="*60)
    
    print("\n[1/2] Processing training data...")
    X_train, y_train, train_failed = load_and_process_dataset(
        csv_path=TRAIN_CSV,
        image_folder=IMAGE_FOLDER
    )
    
    print("\n[2/2] Processing test data...")
    X_test, y_test, test_failed = load_and_process_dataset(
        csv_path=TEST_CSV,
        image_folder=IMAGE_FOLDER
    )
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of classes: {len(set(y_train))}")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Optionally save processed features
    if SAVE_PROCESSED:
        save_processed_data(X_train, y_train, TRAIN_FEATURES_PATH)
        save_processed_data(X_test, y_test, TEST_FEATURES_PATH)
    
    # ========================================
    # TRAIN MODEL
    # ========================================
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    classifier = TrafficSignClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    
    classifier.train(X_train, y_train)
    
    # ========================================
    # EVALUATE MODEL
    # ========================================
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Training set evaluation
    train_acc, _ = classifier.evaluate(X_train, y_train, dataset_name="Training")
    
    # Test set evaluation
    test_acc, _ = classifier.evaluate(X_test, y_test, dataset_name="Test")
    
    # ========================================
    # SAVE MODEL
    # ========================================
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    classifier.save(MODEL_PATH, SCALER_PATH)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Training Accuracy: {train_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")


if __name__ == "__main__":
    main()