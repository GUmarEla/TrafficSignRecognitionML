"""
German Traffic Sign Recognition Benchmark (GTSRB)
Traffic sign classification using traditional ML features
"""

from src.preprocessing import preprocess_image, image_pipeline_minimal
from src.features import feature_extraction
from src.utils import load_and_process_dataset, save_processed_data, load_processed_data
from src.train import TrafficSignClassifier
from src.evaluate import predict_single_image, predict_batch

__version__ = "1.0.0"

__all__ = [
    'preprocess_image',
    'image_pipeline_minimal',
    'feature_extraction',
    'load_and_process_dataset',
    'save_processed_data',
    'load_processed_data',
    'TrafficSignClassifier',
    'predict_single_image',
    'predict_batch'
]