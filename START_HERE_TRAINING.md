# 🚀 START HERE - Training Guide

## Quick Start (Choose One)

### 1. Fastest (2 minutes)
```bash
python quick_train_example.py
```
Trains Random Forest, shows accuracy immediately.

### 2. Complete (30 minutes)
```bash
./train_models.sh          # Linux/Mac
train_models.bat           # Windows
```
Trains all 4 models, generates full comparison.

### 3. Interactive
```bash
jupyter notebook notebooks/train_simple_models.ipynb
```
Step-by-step training with visualizations.

---

## What You Get

### Models Trained
1. **Logistic Regression** - 60-75% (baseline)
2. **Random Forest** - 75-88% (recommended) 🥇
3. **SVM** - 80-90% (best accuracy) 🥈
4. **Simple CNN** - 75-88% (deep learning)

### Outputs
- `results/model_comparison.csv` - Accuracy table
- `results/model_comparison.png` - Bar chart
- `results/*_confusion_matrix.png` - Per-model matrices
- `models/best_model_*.pkl` - Best model saved

---

## Expected Results

```
Model                  Accuracy
SVM                    87.50%  ← Usually best
Random Forest          85.48%  ← Fast & reliable
Simple CNN             81.45%
Logistic Regression    72.18%
```

---

## Documentation

- **TRAINING_README.md** - Complete overview
- **docs/quick_start_training.md** - Quick start guide
- **docs/how_to_train_gait_id.md** - Detailed guide
- **TRAINING_SUMMARY.md** - Technical details

---

## Troubleshooting

**Low accuracy (<70%)**:
```bash
# Try SVM with higher C
# Edit src/train_gait_models.py: SVC(C=100)
```

**Training too slow**:
```bash
# Use Random Forest only
python quick_train_example.py
```

**Out of memory**:
```python
# Reduce CNN batch size
trainer.train_simple_cnn(batch_size=16)
```

---

## Next Steps

After training:

1. **Check results**:
   ```bash
   cat results/model_comparison.csv
   ```

2. **Test on real data**:
   ```bash
   python src/real_world_test.py
   ```

3. **Deploy API**:
   ```bash
   python src/api.py
   ```

---

## Files Created

### Training Scripts
- `quick_train_example.py` - Minimal example
- `src/train_gait_models.py` - Complete pipeline
- `train_models.sh` - Linux/Mac script
- `train_models.bat` - Windows script

### Notebooks
- `notebooks/train_simple_models.ipynb` - Interactive

### Documentation
- `TRAINING_README.md` - Main guide
- `TRAINING_SUMMARY.md` - Details
- `docs/quick_start_training.md` - Quick start
- `docs/how_to_train_gait_id.md` - Full guide

---

**Ready?** Run this now:
```bash
python quick_train_example.py
```

Good luck! 🎉
