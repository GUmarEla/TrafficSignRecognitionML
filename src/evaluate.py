"""
Model evaluation and prediction module
"""
import cv2
import numpy as np
from typing import Union

from src.preprocessing import preprocess_image
from src.features import feature_extraction
from src.train import TrafficSignClassifier


def predict_single_image(
    image_path: str,
    classifier: TrafficSignClassifier
) -> int:
    """
    Predict class for a single image
    
    Args:
        image_path: Path to image file
        classifier: Trained classifier
        
    Returns:
        Predicted class ID
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Preprocess
    processed = preprocess_image(image)
    
    # Extract features
    features = feature_extraction(processed)
    
    # Reshape for prediction (model expects 2D array)
    features = features.reshape(1, -1)
    
    # Predict
    prediction = classifier.predict(features)[0]
    
    return prediction


def predict_batch(
    image_paths: list,
    classifier: TrafficSignClassifier
) -> np.ndarray:
    """
    Predict classes for multiple images
    
    Args:
        image_paths: List of image file paths
        classifier: Trained classifier
        
    Returns:
        Array of predicted class IDs
    """
    features_list = []
    
    for img_path in image_paths:
        try:
            # Load and process
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            processed = preprocess_image(image)
            features = feature_extraction(processed)
            features_list.append(features)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    if len(features_list) == 0:
        raise ValueError("No images were successfully processed")
    
    # Convert to array and predict
    features_array = np.array(features_list)
    predictions = classifier.predict(features_array)
    
    return predictions