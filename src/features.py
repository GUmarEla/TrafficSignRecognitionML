"""
Feature extraction module
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

def feature_extraction(image: np.ndarray) -> np.ndarray:
    """
    Extract HOG + LBP + Color features from 64x64 RGB image
    
    Args:
        image: 64x64 RGB image
        
    Returns:
        Feature vector (~1886 features)
    """
    # Ensure correct size
    if image.shape[:2] != (64, 64):
        image = cv2.resize(image, (64, 64))
    
    # Convert to grayscale for HOG and LBP
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # 1. HOG FEATURES (all of them, no slicing!)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    
    # 2. LBP FEATURES
    radius = 3
    n_points = 8 * radius
    
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_points + 2,
        range=(0, n_points + 2)
    )
    
    lbp_hist = lbp_hist.astype(float)
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)
    
    # 3. COLOR FEATURES (HSV Histograms)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
    
    hist_h = hist_h / (hist_h.sum() + 1e-7)
    hist_s = hist_s / (hist_s.sum() + 1e-7)
    hist_v = hist_v / (hist_v.sum() + 1e-7)
    
    # 4. COMBINE ALL FEATURES
    features = np.concatenate([
        hog_features,  # All HOG features
        lbp_hist,
        hist_h,
        hist_s,
        hist_v
    ])
    
    return features


def extract_features_from_images(images: list) -> np.ndarray:
    """
    Extract features from list of images
    
    Args:
        images: List of preprocessed images
        
    Returns:
        Feature matrix (n_images, n_features)
    """
    from tqdm import tqdm
    
    features_list = []
    
    for img in tqdm(images, desc="Extracting features"):
        features = feature_extraction(img)
        features_list.append(features)
    
    return np.array(features_list)