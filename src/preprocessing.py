"""
Image preprocessing pipeline for traffic sign recognition
"""
import cv2
import numpy as np


def image_pipeline_minimal(image: np.ndarray) -> np.ndarray:
    """
    Minimal preprocessing pipeline: resize to 64x64
    
    Args:
        image: Input image (RGB format)
        
    Returns:
        Resized image (64x64)
    """
    image = cv2.resize(image, (64, 64))
    return image


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Complete preprocessing pipeline for a single image
    
    Args:
        image: Input image in BGR format (as loaded by cv2.imread)
        
    Returns:
        Preprocessed image in RGB format (64x64)
    """
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply minimal pipeline
    processed = image_pipeline_minimal(image)
    
    return processed