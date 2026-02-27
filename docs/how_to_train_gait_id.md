# How to Train Gait ID Model

This guide explains how to train the gait identification model using the cleaned walking data.

## Overview

The gait identification system uses accelerometer data from smartphones to identify individuals based on their unique walking patterns. The training process involves data cleaning, synthetic data generation, and model training.

## Prerequisites

- Python 3.8+
- Required packages (see `requirements.txt`)
- UCI HAR Dataset (should be in `data/datasets/UCI HAR Dataset/`)

## Dataset Structure

The cleaned walking dataset contains:
- **Subjects**: Individual IDs (1-30)
- **Labels**: Activity labels (1 = WALKING)
- **Body Acceleration**: 3-axis accelerometer data (X, Y, Z)
- **Features**: 561 engineered features from the raw signals
- **Sampling Rate**: 50 Hz
- **Window Size**: 128 samples (2.56 seconds)

### Training Set
- 1,226 walking samples
- 21 unique subjects

### Test Set
- 496 walking samples
- 9 unique subjects

## Training Steps

### 1. Data Preparation

Run the gait pipeline notebook to clean and prepare the data:

```bash
jupyter notebook notebooks/gait_pipeline.ipynb
```

The notebook will:
- Load the UCI HAR Dataset
- Filter only WALKING activity (label = 1)
- Save cleaned data to `data/cleaned_walking_data/`

### 2. Synthetic Data Generation (Optional)

To augment the dataset with synthetic samples:

```python
# In the notebook, set:
FORCE_REBUILD_SYNTH = True
DEFAULT_SAMPLES_PER_SUBJECT = 10000  # Adjust as needed
```

The synthetic data generation uses:
- **Time Warping**: Temporal distortion of signals
- **Magnitude Warping**: Amplitude variation
- **3D Rotation**: Simulating different phone orientations
- **Jittering**: Adding realistic noise

### 3. Model Training

Use the training notebook:

```bash
jupyter notebook notebooks/train.ipynb
```

Or run the training script:

```bash
python run.py --mode train
```

#### Training Configuration

Key hyperparameters:
- **Model Architecture**: CNN-LSTM hybrid
- **Input Shape**: (128, 3) for 3-axis accelerometer data
- **Batch Size**: 32first 5 row
- **Learning Rate**: 0.001
- **Epochs**: 50-100
- **Optimizer**: Adam

### 4. Model Evaluation

After training, evaluate the model:

```python
# Load test data
test_data = load_cleaned_split(CLEANED_PATH / 'test')

# Evaluate model
accuracy, confusion_matrix = evaluate_model(model, test_data)
print(f"Test Accuracy: {accuracy:.2%}")
```

## Model Architecture

The gait identification model uses:

1. **Input Layer**: 3-axis accelerometer data (128 timesteps)
2. **Convolutional Layers**: Extract spatial features
3. **LSTM Layers**: Capture temporal patterns
4. **Dense Layers**: Classification
5. **Output Layer**: Softmax for subject identification

## Data Augmentation Techniques

### Time Warping
Applies temporal distortion to simulate different walking speeds:
```python
sigma = 0.15  # Warping intensity
```

### Magnitude Warping
Varies signal amplitude to simulate different phone positions:
```python
sigma = 0.15  # Magnitude variation
```

### 3D Rotation
Simulates different phone orientations:
```python
max_angle = 10  # degrees
```

### Jittering
Adds realistic sensor noise:
```python
sigma = 0.02  # Noise level
```

## Training Tips

1. **Start Small**: Begin with a subset of data to verify the pipeline
2. **Monitor Overfitting**: Use validation split to track performance
3. **Data Balance**: Ensure equal samples per subject
4. **Augmentation**: Use synthetic data to improve generalization
5. **Checkpointing**: Save model checkpoints during training

## Model Deployment

After training, save the model:

```python
# Save model
torch.save(model.state_dict(), 'models/gait_id_production.pth')
```

Deploy using the API:

```bash
python src/api.py
```

## Performance Metrics

Expected performance:
- **Training Accuracy**: 95-98%
- **Test Accuracy**: 85-92%
- **Inference Time**: <100ms per sample

## Troubleshooting

### Low Accuracy
- Increase training epochs
- Add more synthetic data
- Adjust learning rate
- Check data quality

### Overfitting
- Reduce model complexity
- Add dropout layers
- Use more augmentation
- Increase training data

### Memory Issues
- Reduce batch size
- Use gradient accumulation
- Process data in batches

## File Locations

- **Cleaned Data**: `data/cleaned_walking_data/`
- **Synthetic Data**: `data/synthetic_walking_data/`
- **Models**: `models/`
- **Training Notebook**: `notebooks/train.ipynb`
- **Results**: `results/`

## References

- UCI HAR Dataset: [Link](http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
- Project Documentation: `docs/methodology.md`
- Synthetic Data Report: `docs/synthetic_data_generation_report.md`

## Support

For issues or questions:
1. Check the documentation in `docs/`
2. Review the methodology in `docs/methodology.md`
3. Examine the notebook examples in `notebooks/`
