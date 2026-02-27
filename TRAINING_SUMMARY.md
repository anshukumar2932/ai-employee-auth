# Training Files Summary

## 📦 What Was Created

### 1. Main Training Script
**File**: `src/train_gait_models.py`
- Complete training pipeline for all 4 models
- Automatic model comparison and evaluation
- Generates confusion matrices and reports
- Saves best model automatically

**Usage**:
```bash
python src/train_gait_models.py
```

### 2. Interactive Notebook
**File**: `notebooks/train_simple_models.ipynb`
- Step-by-step training with visualizations
- Train models one by one
- Experiment with hyperparameters
- See results immediately

**Usage**:
```bash
jupyter notebook notebooks/train_simple_models.ipynb
```

### 3. Documentation

#### Quick Start Guide
**File**: `docs/quick_start_training.md`
- 3 quick start options
- Expected results
- Model selection guide
- Troubleshooting tips

#### Detailed Training Guide
**File**: `docs/how_to_train_gait_id.md`
- Complete training instructions
- Model architecture details
- Data augmentation techniques
- Performance optimization

#### Training README
**File**: `TRAINING_README.md`
- Project overview
- Model comparison table
- Code examples
- Deployment instructions

### 4. Convenience Scripts

#### Linux/Mac
**File**: `train_models.sh`
```bash
./train_models.sh
```

#### Windows
**File**: `train_models.bat`
```cmd
train_models.bat
```

## 🎯 Models Implemented

### 1. Logistic Regression
- **Purpose**: Baseline sanity check
- **Expected**: 60-75% accuracy
- **Speed**: ⚡⚡⚡ Very Fast
- **Use Case**: Quick validation

### 2. Random Forest 🥇
- **Purpose**: Best simple baseline
- **Expected**: 75-88% accuracy
- **Speed**: ⚡⚡ Fast
- **Use Case**: Production (recommended)

### 3. SVM (RBF Kernel) 🥈
- **Purpose**: Best for small datasets
- **Expected**: 80-90% accuracy
- **Speed**: ⚡ Medium
- **Use Case**: Highest accuracy

### 4. Simple 1D CNN
- **Purpose**: Deep learning baseline
- **Expected**: 75-88% accuracy
- **Speed**: ⚡ Medium (GPU: Fast)
- **Use Case**: Raw signal processing

## 📊 Quick Comparison

| Feature | Logistic | Random Forest | SVM | CNN |
|---------|----------|---------------|-----|-----|
| Accuracy | 60-75% | 75-88% | 80-90% | 75-88% |
| Training Time | 10s | 1-2min | 5-10min | 10-20min |
| Inference Speed | Very Fast | Fast | Medium | Fast |
| Memory Usage | Low | Medium | Medium | High |
| Interpretability | High | High | Low | Low |
| Hyperparameters | Few | Medium | Few | Many |
| Requires Scaling | Yes | No | Yes | Yes |
| Works on Features | ✅ | ✅ | ✅ | ❌ |
| Works on Raw Signals | ❌ | ❌ | ❌ | ✅ |

## 🚀 Recommended Workflow

### For Beginners
1. Run `train_models.sh` or `train_models.bat`
2. Check `results/model_comparison.csv`
3. Use the best model

### For Experimentation
1. Open `notebooks/train_simple_models.ipynb`
2. Train models step by step
3. Experiment with hyperparameters
4. Visualize results interactively

### For Production
1. Train all models with `src/train_gait_models.py`
2. Select best model (usually SVM or Random Forest)
3. Save model: Already done automatically
4. Deploy with `src/api.py`

## 📈 Expected Output

### Console Output
```
==========================================
Training Logistic Regression (Baseline)
==========================================
Accuracy: 0.7218
F1 Score: 0.7156

==========================================
Training Random Forest (Recommended Baseline)
==========================================
Accuracy: 0.8548
F1 Score: 0.8501

==========================================
Training SVM with RBF Kernel (Best for Small Data)
==========================================
Accuracy: 0.8750
F1 Score: 0.8723

==========================================
Training Simple 1D CNN (Deep Learning)
==========================================
Epoch 5/50 - Train Loss: 1.2345, Train Acc: 75.23%, Test Acc: 72.18%
...
Best Test Accuracy: 0.8145

==========================================
MODEL COMPARISON
==========================================
Model                  Accuracy    F1 Score
SVM                    0.8750      0.8723
Random Forest          0.8548      0.8501
Simple CNN             0.8145      0.8098
Logistic Regression    0.7218      0.7156

Best Model: SVM with 0.8750 (87.50%) accuracy
```

### Generated Files
```
results/
├── model_comparison.csv
├── model_comparison.png
├── Logistic_Regression_confusion_matrix.png
├── Random_Forest_confusion_matrix.png
├── SVM_confusion_matrix.png
├── Simple_CNN_confusion_matrix.png
├── cnn_training_curves.png
├── Logistic_Regression_report.txt
├── Random_Forest_report.txt
├── SVM_report.txt
└── Simple_CNN_report.txt

models/
├── best_model_metadata.json
├── simple_cnn_best.pth
└── best_model_svm.pkl (or random_forest.pkl)
```

## 🎓 Key Insights

### Why These Models?

1. **Logistic Regression**: Establishes baseline. If it performs well (>70%), features are discriminative.

2. **Random Forest**: 
   - No hyperparameter tuning needed
   - Robust to noise
   - Provides feature importance
   - Fast inference

3. **SVM**:
   - Proven best for HAR (Human Activity Recognition)
   - Excellent with 561 engineered features
   - Handles non-linear patterns well
   - Usually achieves highest accuracy

4. **Simple CNN**:
   - End-to-end learning from raw signals
   - No manual feature engineering
   - Better for mobile deployment
   - Requires more data (use synthetic augmentation)

### Dataset Characteristics

- **Size**: 1,226 training samples (small)
- **Classes**: 21 subjects (multi-class)
- **Features**: 561 (high-dimensional)
- **Imbalance**: Moderate (some subjects have more samples)

**Implication**: Classical ML (especially SVM) often outperforms deep learning on this dataset size.

### When to Use Each Model

**Use Random Forest when**:
- You need fast training and inference
- You want feature importance
- You need a robust baseline
- You're deploying to production

**Use SVM when**:
- You want maximum accuracy
- Training time is not critical
- You have the 561 features
- Dataset is small (<10k samples)

**Use CNN when**:
- You have raw signals only
- You plan to use synthetic data
- You're deploying to mobile
- You have GPU available

**Use Logistic Regression when**:
- You need a quick sanity check
- You want the fastest possible model
- You need a baseline to beat

## 🔧 Customization

### Modify Hyperparameters

Edit `src/train_gait_models.py`:

```python
# Random Forest
RandomForestClassifier(
    n_estimators=200,      # Try 100, 200, 500
    max_depth=None,        # Try 10, 20, None
    min_samples_split=5,   # Try 2, 5, 10
)

# SVM
SVC(
    C=10,                  # Try 1, 10, 100
    gamma='scale',         # Try 'scale', 'auto', 0.01
)

# CNN
train_simple_cnn(
    epochs=50,             # Try 30, 50, 100
    batch_size=32,         # Try 16, 32, 64
    lr=0.001              # Try 0.0001, 0.001, 0.01
)
```

### Add New Models

Add to `GaitTrainer` class:

```python
def train_xgboost(self):
    from xgboost import XGBClassifier
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1
    )
    model.fit(self.X_train_features, self.y_train)
    # ... evaluation code
```

## 📚 Additional Resources

- **UCI HAR Dataset**: http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
- **Scikit-learn Docs**: https://scikit-learn.org/
- **PyTorch Docs**: https://pytorch.org/docs/
- **Project Methodology**: `docs/methodology.md`

## ❓ FAQ

**Q: Which file should I run first?**  
A: Run `train_models.sh` (Linux/Mac) or `train_models.bat` (Windows), or use the notebook.

**Q: How long does training take?**  
A: 10-30 minutes total. Logistic Regression: 10s, Random Forest: 1-2min, SVM: 5-10min, CNN: 10-20min.

**Q: Which model should I use in production?**  
A: Usually SVM for accuracy or Random Forest for speed. Check `results/model_comparison.csv`.

**Q: Can I skip some models?**  
A: Yes! Edit `src/train_gait_models.py` and comment out models you don't want.

**Q: How do I improve accuracy?**  
A: 1) Use SVM, 2) Generate synthetic data, 3) Try ensemble methods, 4) Tune hyperparameters.

**Q: Why is SVM so slow?**  
A: SVM is O(n²) to O(n³). With 1,226 samples, 5-10 minutes is normal.

**Q: Can I use GPU?**  
A: Yes, for CNN only. PyTorch will automatically use GPU if available.

**Q: What if I get low accuracy (<70%)?**  
A: Check data loading, try SVM with C=100, generate synthetic data, or verify labels.

## 🎉 Success Criteria

Your training is successful if:
- ✅ All models train without errors
- ✅ SVM or Random Forest achieves >80% accuracy
- ✅ Confusion matrices show diagonal patterns
- ✅ Best model is saved automatically
- ✅ Results are reproducible

## 🚀 Next Steps

After successful training:

1. **Test on Real Data**:
   ```bash
   python src/real_world_test.py
   ```

2. **Deploy API**:
   ```bash
   python src/api.py
   ```

3. **Generate More Synthetic Data**:
   ```bash
   jupyter notebook notebooks/gait_pipeline.ipynb
   # Set FORCE_REBUILD_SYNTH = True
   ```

4. **Try Ensemble Methods**:
   Combine best models for even better accuracy

---

**Ready to start?** Choose your path:

- **Easiest**: `./train_models.sh` or `train_models.bat`
- **Interactive**: `jupyter notebook notebooks/train_simple_models.ipynb`
- **Advanced**: `python src/train_gait_models.py`

Good luck! 🚀
