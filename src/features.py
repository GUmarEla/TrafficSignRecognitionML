"""
Feature extraction module for traffic sign recognition
Extracts HOG, LBP, and color histogram features
"""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


def extract_hog_features(gray: np.ndarray) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features
    
    Args:
        gray: Grayscale image (64x64)
        
    Returns:
        HOG feature vector (~1764 features)
    """
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    return hog_features


def extract_lbp_features(gray: np.ndarray) -> np.ndarray:
    """
    Extract LBP (Local Binary Pattern) features
    
    Args:
        gray: Grayscale image (64x64)
        
    Returns:
        LBP histogram (26 features)
    """
    radius = 3
    n_points = 8 * radius  # 24 points

    lbp = local_binary_pattern(
        gray,
        n_points,
        radius,
        method='uniform'
    )

    # Histogram of LBP patterns
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_points + 2,
        range=(0, n_points + 2)
    )

    # Normalize
    lbp_hist = lbp_hist.astype(float)
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-7)

    return lbp_hist


def extract_color_features(image: np.ndarray) -> np.ndarray:
    """
    Extract color histogram features from HSV color space
    
    Args:
        image: RGB image (64x64)
        
    Returns:
        Concatenated HSV histograms (96 features: 32+32+32)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Calculate histograms for each channel
    hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

    # Normalize each histogram
    hist_h = hist_h / (hist_h.sum() + 1e-7)
    hist_s = hist_s / (hist_s.sum() + 1e-7)
    hist_v = hist_v / (hist_v.sum() + 1e-7)

    # Concatenate
    color_features = np.concatenate([hist_h, hist_s, hist_v])
    
    return color_features


def feature_extraction(image: np.ndarray) -> np.ndarray:
    """
    Extract all features from 64x64 RGB image
    
    Args:
        image: Preprocessed RGB image (64x64)
        
    Returns:
        Feature vector (1886 features total: 1764 HOG + 26 LBP + 96 Color)
    """
    # Ensure correct size (defensive programming)
    if image.shape[:2] != (64, 64):
        image = cv2.resize(image, (64, 64))

    # Convert to grayscale for HOG and LBP
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Extract all feature types
    hog_features = extract_hog_features(gray)
    lbp_features = extract_lbp_features(gray)
    color_features = extract_color_features(image)

    # Combine all features
    features = np.concatenate([
        hog_features,    # ~1764 features
        lbp_features,    # 26 features
        color_features   # 96 features
    ])

    return features