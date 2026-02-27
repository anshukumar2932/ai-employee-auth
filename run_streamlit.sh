#!/bin/bash

echo "🚀 Starting Stark Industries Gait Authentication System..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements_streamlit.txt
fi

# Check if model exists
if [ ! -f "models/best_model_logistic_regression.pkl" ]; then
    echo "⚠️  Model not found. Training model first..."
    python src/train_gait_models.py
fi

echo ""
echo "✅ All checks passed!"
echo "🌐 Opening Streamlit app..."
echo ""

# Run streamlit
streamlit run app.py
