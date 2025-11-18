"""
Model training module
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def train_model(X_train: np.ndarray, y_train: np.ndarray, config: dict):
    """
    Train Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration dictionary
        
    Returns:
        model: Trained model
        scaler: Fitted scaler
    """
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=config.get('n_estimators', 200),
        max_depth=config.get('max_depth', None),
        min_samples_split=config.get('min_samples_split', 2),
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Training accuracy
    train_pred = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    
    return model, scaler