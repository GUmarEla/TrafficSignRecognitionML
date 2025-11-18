"""
Model evaluation module
"""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, scaler, X_test: np.ndarray, y_test: np.ndarray):
    """
    Evaluate trained model
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_test: Test features
        y_test: Test labels
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    print("\n" + "="*50)
    print("EVALUATING MODEL")
    print("="*50)
    
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - Accuracy: {test_acc:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    return {
        'accuracy': test_acc,
        'predictions': y_pred,
        'confusion_matrix': cm
    }