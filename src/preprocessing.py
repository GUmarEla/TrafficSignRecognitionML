"""
Image preprocessing pipeline
"""
import cv2
import numpy as np

def image_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Minimal effective pipeline
    Args:
        image: RGB image (any size)
    Returns:
        Processed 64x64 RGB image
    """
    # Resize to standard size
    image = cv2.resize(image, (64, 64))
    
    # Illumination normalization (CLAHE on LAB)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    image = cv2.merge([l, a, b])
    image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    
    return image


def load_and_preprocess_dataset(csv_path: str, image_folder: str):
    """
    Load and preprocess all images from dataset
    
    Args:
        csv_path: Path to Train.csv or Test.csv
        image_folder: Base folder containing images
        
    Returns:
        processed_images: List of preprocessed images
        labels: List of class labels
        failed_images: List of failed image paths
    """
    import pandas as pd
    from tqdm import tqdm
    import os
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    
    processed_images = []
    labels = []
    failed_images = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        try:
            img_path = os.path.join(image_folder, row['Path'])
            image = cv2.imread(img_path)
            
            if image is None:
                failed_images.append(img_path)
                continue
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply pipeline
            processed = image_pipeline(image)
            
            processed_images.append(processed)
            labels.append(row['ClassId'])
            
        except Exception as e:
            failed_images.append(f"{img_path}: {str(e)}")
            continue
    
    print(f"Processed: {len(processed_images)}, Failed: {len(failed_images)}")
    
    return processed_images, labels, failed_images