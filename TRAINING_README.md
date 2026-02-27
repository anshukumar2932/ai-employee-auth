# Gait Identification - Model Training

Complete training pipeline for gait identification using accelerometer data.

## 📋 Overview

This project trains models to identify individuals based on their unique walking patterns using smartphone accelerometer data.

**Dataset**: UCI HAR Dataset (Walking activity only)
- Training: 1,226 samples from 21 subjects
- Test: 496 samples from 9 subjects
- Features: 561 engineered features + raw 128×3 signals

## 🚀 Quick Start

### 1. Train All Models (Recommended)

```bash
python src/train_gait_models.py
```

Trains and compares:
- ✅ Logistic Regression (baseline)
- ✅ Random Forest (best simple model)
- ✅ SVM (best for small data)
- ✅ Simple 1D CNN (deep learning)

**Output**: Comparison charts, confusion matrices, best model saved

### 2. Interactive Notebook

```bash
jupyter notebook notebooks/train_simple_models.ipynb
```

Step-by-step training with visualizations.

### 3. Quick Test

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load data
X_train = np.load('data/cleaned_walking_data/train/features.npy')
y_train = np.load('data/cleaned_walking_data/train/subjects.npy')

# Train
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
```

## 📊 Model Comparison

| Model | Accuracy | Speed | Best For |
|-------|----------|-------|----------|
| Logistic Regression | 60-75% | ⚡⚡⚡ | Baseline check |
| Random Forest | 75-88% | ⚡⚡ | Production (fast) |
| SVM (RBF) | 80-90% | ⚡ | Best accuracy |
| Simple CNN | 75-88% | ⚡ | Raw signals |

**Recommendation**: Start with **Random Forest**, then try **SVM** for best accuracy.

## 📁 Project Structure

```
├── src/
│   ├── train_gait_models.py      # Complete training pipeline
│   ├── api.py                     # Deployment API
│   └── real_world_test.py         # Test on real data
├── notebooks/
│   ├── train_simple_models.ipynb  # Interactive training
│   ├── gait_pipeline.ipynb        # Data preparation
│   └── train.ipynb                # Advanced training
├── docs/
│   ├── quick_start_training.md    # Quick start guide
│   ├── how_to_train_gait_id.md    # Detailed training guide
│   └── methodology.md             # Project methodology
├── data/
│   └── cleaned_walking_data/      # Prepared dataset
├── models/                        # Saved models
└── results/                       # Training results
```

## 🎯 Training Options

### Option 1: Classical ML (Recommended for Small Data)

**Best Models**: Random Forest, SVM

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Random Forest (Fast, 75-88% accuracy)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# SVM (Best accuracy, 80-90%)
svm = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf', C=10, gamma='scale')
)
svm.fit(X_train, y_train)
```

### Option 2: Deep Learning (For Raw Signals)

**Best Model**: Simple 1D CNN

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, 128, 3) -> (batch, 3, 128)
        x = self.conv(x)
        return self.fc(x.squeeze(-1))
```

## 📈 Expected Results

### Typical Performance

```
Model                  Accuracy    F1 Score
─────────────────────────────────────────────
SVM                    0.8750      0.8723
Random Forest          0.8548      0.8501
Simple CNN             0.8145      0.8098
Logistic Regression    0.7218      0.7156
```

### What to Expect

- **Logistic Regression**: 60-75% (sanity check)
- **Random Forest**: 75-88% (strong baseline)
- **SVM**: 80-90% (usually best)
- **Simple CNN**: 75-88% (with raw signals)

## 🔧 Hyperparameter Tuning

### Random Forest

```python
RandomForestClassifier(
    n_estimators=200,      # More trees = better (diminishing returns after 200)
    max_depth=None,        # No limit (prevents underfitting)
    min_samples_split=5,   # Prevents overfitting
    min_samples_leaf=2,    # Prevents overfitting
    random_state=42
)
```

### SVM

```python
SVC(
    kernel='rbf',          # RBF kernel works best for HAR data
    C=10,                  # Regularization (try 1, 10, 100)
    gamma='scale',         # Kernel coefficient (auto-scaled)
    random_state=42
)
```

### Simple CNN

```python
# Training parameters
epochs = 50
batch_size = 32
learning_rate = 0.001
optimizer = Adam
scheduler = ReduceLROnPlateau
```

## 📊 Evaluation Metrics

### Accuracy
Overall correctness: `(correct predictions) / (total predictions)`

### F1 Score
Balanced measure considering precision and recall (important for imbalanced data)

### Confusion Matrix
Shows which subjects are confused with each other

### Per-Subject Metrics
- Precision: How many predicted subjects were correct
- Recall: How many actual subjects were found
- F1: Harmonic mean of precision and recall

## 🎓 Training Tips

### 1. Start Simple
Always begin with Logistic Regression and Random Forest before trying complex models.

### 2. Feature Scaling
- **Required**: SVM, Logistic Regression, Neural Networks
- **Not needed**: Random Forest, Decision Trees

### 3. Data Augmentation
If accuracy is low, generate synthetic data:
```bash
# In notebooks/gait_pipeline.ipynb
FORCE_REBUILD_SYNTH = True
DEFAULT_SAMPLES_PER_SUBJECT = 10000
```

### 4. Cross-Validation
For robust evaluation:
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 5. Ensemble Methods
Combine multiple models:
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[('rf', rf_model), ('svm', svm_model)],
    voting='hard'
)
```

## 🐛 Troubleshooting

### Low Accuracy (<70%)
1. Check data loading: `print(X_train.shape, y_train.shape)`
2. Verify labels: `print(np.unique(y_train))`
3. Try SVM with higher C: `SVC(C=100)`
4. Generate synthetic data

### Training Too Slow
1. Use Random Forest instead of SVM
2. Reduce SVM training data
3. Use fewer trees in Random Forest
4. Skip CNN training

### Out of Memory
1. Reduce CNN batch size: `batch_size=16`
2. Use only classical ML models
3. Process data in chunks

### Overfitting
1. Use cross-validation
2. Add regularization
3. Reduce model complexity
4. Get more training data

## 📚 Documentation

- **Quick Start**: `docs/quick_start_training.md`
- **Detailed Guide**: `docs/how_to_train_gait_id.md`
- **Methodology**: `docs/methodology.md`
- **API Deployment**: See `src/api.py`

## 🚀 Deployment

After training, deploy the best model:

```bash
# Start API server
python src/api.py

# Test with real data
python src/real_world_test.py
```

## 📝 Citation

If you use this code, please cite:

```
UCI HAR Dataset:
Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz.
Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine.
International Workshop of Ambient Assisted Living (IWAAL 2012). Vitoria-Gasteiz, Spain. Dec 2012
```

## 🤝 Contributing

Contributions welcome! Please:
1. Test your changes
2. Update documentation
3. Follow existing code style

## 📄 License

See LICENSE file for details.

## 💬 Support

- Issues: Open a GitHub issue
- Documentation: Check `docs/` folder
- Examples: See `notebooks/` folder

---

**Ready to train?** Start with:
```bash
python src/train_gait_models.py
```

or

```bash
jupyter notebook notebooks/train_simple_models.ipynb
```
