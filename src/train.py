"""
Model training and evaluation module
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from typing import Tuple, Optional


class TrafficSignClassifier:
    """
    Traffic sign classifier using Random Forest with feature scaling
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize classifier
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples required to split a node
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=1
        )
        
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> None:
        """
        Train the classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("Training Random Forest classifier...")
        self.model.fit(X_train_scaled, y_train)
        print("Training complete!")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y_true: np.ndarray,
        dataset_name: str = "Test"
    ) -> Tuple[float, str]:
        """
        Evaluate model performance
        
        Args:
            X: Features
            y_true: True labels
            dataset_name: Name for printing (e.g., "Test", "Train")
            
        Returns:
            accuracy: Classification accuracy
            report: Classification report string
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        
        print(f"\n{dataset_name} Set Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(report)
        
        return accuracy, report
    
    def save(self, model_path: str, scaler_path: str) -> None:
        """
        Save trained model and scaler
        
        Args:
            model_path: Path to save model
            scaler_path: Path to save scaler
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load(self, model_path: str, scaler_path: str) -> None:
        """
        Load trained model and scaler
        
        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")