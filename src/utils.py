"""
Utility functions
"""
import yaml
import numpy as np
import pickle
import os

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_features(features: np.ndarray, labels: np.ndarray, output_path: str):
    """Save features and labels to .npz file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, features=features, labels=labels)
    print(f"Saved features to {output_path}")


def load_features(features_path: str) -> tuple:
    """Load features and labels from .npz file"""
    data = np.load(features_path)
    return data['features'], data['labels']


def save_model(model, scaler, model_path: str):
    """Save trained model and scaler"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    print(f"Saved model to {model_path}")


def load_model(model_path: str):
    """Load trained model and scaler"""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['model'], data['scaler']


def download_dataset_from_kaggle(dataset_name: str, output_dir: str):
    """
    Download dataset from Kaggle using opendatasets
    
    Args:
        dataset_name: Kaggle dataset URL or name
        output_dir: Where to save the dataset
    """
    import opendatasets as od
    
    print(f"Downloading dataset from Kaggle...")
    od.download(dataset_name, data_dir=output_dir)
    print(f"Dataset downloaded to {output_dir}")