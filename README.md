# Traffic Sign Recognition

Classical ML approach using HOG + LBP + Color features with Random Forest classifier.

## Accuracy: 92%

## Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Option 1: Run complete pipeline
```bash
python scripts/train_model.py
```

### Option 2: Run step-by-step
```python
# 1. Preprocess
from src.preprocessing import load_and_preprocess_dataset
images, labels, _ = load_and_preprocess_dataset('data/raw/Train.csv', 'data/raw')

# 2. Extract features
from src.features import extract_features_from_images
features = extract_features_from_images(images)

# 3. Train
from src.train import train_model
model, scaler = train_model(features, labels, config)

# 4. Evaluate
from src.evaluate import evaluate_model
results = evaluate_model(model, scaler, X_test, y_test)
```

## Project Structure
```
├── configs/          # Configuration files
├── data/            # Dataset (gitignored)
├── models/          # Trained models (gitignored)
├── scripts/         # Executable scripts
├── src/             # Source code modules
└── tests/           # Unit tests
```