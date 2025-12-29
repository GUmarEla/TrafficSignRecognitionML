"""
Utility functions for dataset loading and processing
"""
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List

from src.preprocessing import preprocess_image
from src.features import feature_extraction


def load_and_process_dataset(
    csv_path: str, 
    image_folder: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load images from CSV and extract features
    
    Args:
        csv_path: Path to CSV file containing image paths and labels
        image_folder: Root folder containing images
        
    Returns:
        features: Array of feature vectors (n_samples, n_features)
        labels: Array of class labels (n_samples,)
        failed_images: List of images that failed to process
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    features_list = []
    labels_list = []
    failed_images = []

    # Process each image
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        try:
            # Construct image path
            img_path = os.path.join(image_folder, row['Path'])
            
            # Load image
            image = cv2.imread(img_path)

            if image is None:
                failed_images.append(img_path)
                continue

            # Preprocess image
            processed = preprocess_image(image)

            # Extract features
            features = feature_extraction(processed)

            # Store results
            features_list.append(features)
            labels_list.append(row['ClassId'])

        except Exception as e:
            failed_images.append(f"{img_path}: {str(e)}")
            continue

    # Summary
    print(f"Successfully processed: {len(features_list)}")
    print(f"Failed: {len(failed_images)}")

    if len(features_list) == 0:
        raise ValueError("No images were successfully processed!")

    # Convert to numpy arrays
    features = np.array(features_list)
    labels = np.array(labels_list)

    return features, labels, failed_images


def save_processed_data(
    features: np.ndarray, 
    labels: np.ndarray, 
    output_path: str
) -> None:
    """
    Save processed features and labels to disk
    
    Args:
        features: Feature array
        labels: Label array
        output_path: Path to save .npz file
    """
    np.savez(output_path, features=features, labels=labels)
    print(f"Saved processed data to {output_path}")


def load_processed_data(input_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load previously processed features and labels
    
    Args:
        input_path: Path to .npz file
        
    Returns:
        features: Feature array
        labels: Label array
    """
    data = np.load(input_path)
    features = data['features']
    labels = data['labels']
    print(f"Loaded processed data from {input_path}")
    return features, labels