# German Traffic Sign Recognition Benchmark (GTSRB)

Traffic sign classification using traditional machine learning with hand-crafted features (HOG, LBP, Color Histograms) and Random Forest classifier.

**Test Accuracy: 93%**

## Project Structure

```
TRAFFIC SIGN RECOGNITION/
├── configs/
│   └── config.yaml              # Configuration file (optional)
├── data/
│   └── .keep                    # Placeholder for data directory
├── scripts/
│   └── train_model.py           # Main training script
├── src/
│   ├── __init__.py              # Package initialization
│   ├── preprocessing.py         # Image preprocessing pipeline
│   ├── features.py              # Feature extraction (HOG, LBP, Color)
│   ├── utils.py                 # Dataset loading utilities
│   ├── train.py                 # Model training and classifier class
│   └── evaluate.py              # Prediction and evaluation
├── .gitignore
├── README.md
├── requirements.txt             # Python dependencies
└── setup.py                     # Package setup file
```

## Quick Start (Google Colab)

### 1. Clone/Upload Project

```python
# In Colab, upload your project or clone from GitHub
!git clone https://github.com/yourusername/traffic-sign-recognition.git
%cd traffic-sign-recognition
```

### 2. Install Dependencies

```python
!pip install -r requirements.txt
```

### 3. Upload Your Dataset

Upload your GTSRB dataset to Colab or mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Update Paths in `train_model.py`

Edit the configuration section in `scripts/train_model.py`:

```python
TRAIN_CSV = '/content/gtsrb-german-traffic-sign/Train.csv'
TEST_CSV = '/content/gtsrb-german-traffic-sign/Test.csv'
IMAGE_FOLDER = '/content/gtsrb-german-traffic-sign'
```

### 5. Train the Model

```python
!python scripts/train_model.py
```

## Features Extracted

The model uses **1,886 hand-crafted features**:

1. **HOG Features** (~1,764): Captures shape and edge information
2. **LBP Features** (26): Local texture patterns
3. **Color Histograms** (96): HSV color distribution (32 bins × 3 channels)

## Model Architecture

- **Classifier**: Random Forest (200 trees)
- **Feature Scaling**: StandardScaler
- **Input**: 64×64 RGB images
- **Output**: 43 traffic sign classes

## Usage Examples

### Train from Scratch

```python
from src.utils import load_and_process_dataset
from src.train import TrafficSignClassifier

# Load data
X_train, y_train, _ = load_and_process_dataset(
    csv_path='path/to/Train.csv',
    image_folder='path/to/images'
)

# Train
classifier = TrafficSignClassifier(n_estimators=200)
classifier.train(X_train, y_train)
classifier.save('model.pkl', 'scaler.pkl')
```

### Make Predictions

```python
from src.train import TrafficSignClassifier
from src.evaluate import predict_single_image

# Load trained model
classifier = TrafficSignClassifier()
classifier.load('model.pkl', 'scaler.pkl')

# Predict
class_id = predict_single_image('path/to/image.jpg', classifier)
print(f"Predicted class: {class_id}")
```

## Dataset Format

Your CSV should have these columns:
- `Path`: Relative path to image (e.g., "Train/0/00000.png")
- `ClassId`: Integer class label (0-42)

Example:
```csv
Path,ClassId
Train/0/00000.png,0
Train/0/00001.png,0
Train/1/00000.png,1
```

## Configuration

Modify hyperparameters in `scripts/train_model.py`:

```python
classifier = TrafficSignClassifier(
    n_estimators=200,        # Number of trees
    max_depth=None,          # Tree depth (None = unlimited)
    min_samples_split=2,     # Min samples to split
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)
```

## Expected Results

- **Training Accuracy**: ~98-99%
- **Test Accuracy**: ~93%
- **Training Time**: ~5-10 minutes (on Colab with CPU)

## Troubleshooting

### "No images processed" error
- Check that `Path` column in CSV matches actual file locations
- Verify `IMAGE_FOLDER` is the correct root directory

### Memory issues
- Process data in batches
- Reduce `n_estimators` or `max_depth`

### Import errors
- Ensure all dependencies are installed: `!pip install -r requirements.txt`
- Make sure you're running from project root directory

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Contact

EL ALLAM OMAR - omar.elallam19@gmail.com
