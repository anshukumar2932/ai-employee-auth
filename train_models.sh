#!/bin/bash

# Gait Identification Model Training Script
# This script trains all models and generates comparison reports

echo "=========================================="
echo "Gait Identification Model Training"
echo "=========================================="
echo ""

# Check if data exists
if [ ! -d "data/cleaned_walking_data" ]; then
    echo "❌ Error: Cleaned data not found!"
    echo "Please run the data preparation notebook first:"
    echo "  jupyter notebook notebooks/gait_pipeline.ipynb"
    exit 1
fi

# Create necessary directories
mkdir -p results
mkdir -p models

echo "✅ Data found"
echo "✅ Directories created"
echo ""

# Check Python dependencies
echo "Checking dependencies..."
python -c "import sklearn, torch, numpy, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
else
    echo "✅ All dependencies installed"
fi

echo ""
echo "=========================================="
echo "Starting Model Training"
echo "=========================================="
echo ""
echo "This will train 4 models:"
echo "  1. Logistic Regression (baseline)"
echo "  2. Random Forest (best simple model)"
echo "  3. SVM (best for small data)"
echo "  4. Simple 1D CNN (deep learning)"
echo ""
echo "Estimated time: 10-30 minutes"
echo ""

# Run training
python src/train_gait_models.py

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Training Complete!"
    echo "=========================================="
    echo ""
    echo "Results saved to:"
    echo "  📊 results/model_comparison.csv"
    echo "  📊 results/model_comparison.png"
    echo "  📊 results/*_confusion_matrix.png"
    echo "  💾 models/best_model_metadata.json"
    echo ""
    echo "To view results:"
    echo "  cat results/model_comparison.csv"
    echo "  open results/model_comparison.png"
    echo ""
else
    echo ""
    echo "❌ Training failed. Check the error messages above."
    exit 1
fi
