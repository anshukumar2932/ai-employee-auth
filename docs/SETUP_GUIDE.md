# Complete Setup Guide
## AI-Powered Contactless Employee Security System

This guide will help you set up and run the gait-based authentication system in 5 minutes.

---

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **GPU**: Optional but recommended (NVIDIA RTX 3050+ or equivalent)
- **Storage**: 2GB free space

### Check Python Version
```bash
python --version
# or
python3 --version
```

If Python is not installed, download from: https://www.python.org/downloads/

---

## Quick Start (5 Minutes)

### Step 1: Clone Repository
```bash
git clone <your-repository-url>
cd ai-employee-auth
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- torch (PyTorch for deep learning)
- numpy (numerical operations)
- pandas (data manipulation)
- scikit-learn (machine learning utilities)
- scipy (scientific computing)
- matplotlib & seaborn (visualization)
- streamlit (web application)
- plotly (interactive charts)
- flask (API server)
- jupyter (notebooks)

### Step 4: Run Streamlit Application
```bash
streamlit run app.py
```

The application will open in your browser at: http://localhost:8501

**That's it!** You're ready to use the system.

---

## Detailed Setup Options

### Option 1: Direct Installation (Fastest)
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Development Setup
```bash
# Install in editable mode
pip install -e .

# Install development dependencies
pip install jupyter pytest black flake8

# Run Jupyter notebooks
jupyter notebook
```

### Option 3: Docker Setup (Coming Soon)
```bash
docker build -t gait-auth .
docker run -p 8501:8501 gait-auth
```

---

## Running Different Components

### 1. Streamlit Web Application (Recommended)
```bash
streamlit run app.py
```
- **URL**: http://localhost:8501
- **Features**: Full UI, authentication, analytics, demo mode
- **Best for**: Non-technical users, demonstrations

### 2. Flask API Server
```bash
python src/api.py
```
- **URL**: http://localhost:5000
- **Endpoints**: /authenticate, /batch_authenticate, /health, /stats
- **Best for**: Programmatic access, integration with other systems

### 3. Jupyter Notebooks
```bash
jupyter notebook
```
Then open:
- `notebooks/gait_pipeline.ipynb` - Data cleaning & synthetic generation
- `notebooks/train.ipynb` - Model training

### 4. Command-Line Scripts
```bash
# Train models
python src/train_gait_models.py

# Test real-world data
python src/real_world_test.py --csv_file data.csv

# Quick execution
python run.py
```

---

## Using the Streamlit Application

### Home Page
- View system overview and statistics
- See recent authentication attempts
- Check model accuracy and performance

### Authentication Page
**Upload Data Tab**:
1. Click "Browse files" or drag & drop CSV file
2. CSV should have columns: `time`, `accel_x`, `accel_y`, `accel_z`
3. Click "Authenticate" button
4. View results with confidence score

**Demo Mode Tab**:
1. Select a sample from the slider
2. Click "Test Authentication"
3. See predicted ID, confidence, and match result

### Analytics Page
- View access logs and statistics
- See success/failure rates
- Analyze user patterns
- Download access logs as CSV

### Real-World Test Page
1. Collect data using Physics Toolbox Sensor Suite app
2. Export as CSV
3. Upload to this page
4. View authentication results

### About Page
- Project overview and methodology
- Performance metrics
- Future enhancements
- Team information

---

## Collecting Real-World Data

### Using Physics Toolbox Sensor Suite

#### Step 1: Install App
- **Android**: https://play.google.com/store/apps/details?id=com.chrystianvieyra.physicstoolboxsuite
- **iOS**: Search "Physics Toolbox" in App Store

#### Step 2: Configure Settings
1. Open app
2. Select "Accelerometer"
3. Set sampling rate to 50Hz
4. Enable data recording

#### Step 3: Collect Data
1. Place phone in pocket or hold naturally
2. Walk normally for 5-10 seconds
3. Stop recording
4. Export as CSV

#### Step 4: Test in Application
1. Upload CSV to "Real-World Test" page
2. View authentication results
3. Check confidence scores

### Expected CSV Format
```csv
time,accel_x,accel_y,accel_z
0.00,0.12,9.81,0.05
0.02,0.15,9.79,0.07
0.04,0.18,9.83,0.04
...
```

---

## Training Your Own Model

### Quick Training (2 minutes)
```bash
# Train Random Forest only
python quick_train_example.py
```

### Complete Training (30 minutes)
```bash
# Train all models (Logistic Regression, Random Forest, SVM, CNN-LSTM)
python src/train_gait_models.py
```

### Interactive Training
```bash
# Open Jupyter notebook
jupyter notebook notebooks/train.ipynb

# Run all cells to train CNN-LSTM-Attention model
```

### Training Output
Models will be saved to `models/` directory:
- `best_model_logistic_regression.pkl`
- `best_model_random_forest.pkl`
- `gait_id_production.pth` (CNN-LSTM-Attention)

---

## Troubleshooting

### Issue: "pip: command not found"
**Solution**: Use `pip3` instead of `pip`
```bash
pip3 install -r requirements.txt
```

### Issue: "streamlit: command not found"
**Solution**: Ensure virtual environment is activated or use full path
```bash
python -m streamlit run app.py
```

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Install PyTorch manually
```bash
pip install torch torchvision torchaudio
```

### Issue: GPU not detected
**Solution**: Install CUDA-enabled PyTorch
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "Port 8501 already in use"
**Solution**: Use a different port
```bash
streamlit run app.py --server.port 8502
```

### Issue: Model files not found
**Solution**: Download pre-trained models or train new ones
```bash
# Train new models
python src/train_gait_models.py
```

### Issue: Out of memory during training
**Solution**: Reduce batch size in training script
```python
# In src/train_gait_models.py
BATCH_SIZE = 32  # Reduce from 64
```

---

## Testing the Installation

### Quick Test
```bash
# Run Streamlit app
streamlit run app.py

# In browser, go to "Demo Mode" tab
# Click "Test Authentication"
# Should see results without errors
```

### Full Test
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_api.py
```

---

## Project Structure

```
ai-employee-auth/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
├── run.py                      # Quick execution script
│
├── notebooks/                  # Jupyter notebooks
│   ├── gait_pipeline.ipynb    # Data cleaning & synthetic generation
│   └── train.ipynb            # Model training
│
├── src/                        # Source code
│   ├── api.py                 # Flask API server
│   ├── train_gait_models.py   # Training pipeline
│   ├── real_world_test.py     # Real-world testing
│   ├── data_cleaner.py        # Data preprocessing
│   └── mobile_processor.py    # Mobile data processing
│
├── models/                     # Trained models
│   ├── gait_id_production.pth # Main CNN-LSTM model
│   ├── best_model_*.pkl       # Traditional ML models
│   └── best_model_metadata.json
│
├── data/                       # Datasets
│   ├── cleaned_walking_data/  # Cleaned UCI HAR data
│   ├── synthetic_walking_data/ # Synthetic data
│   ├── real_world_samples/    # Real-world test data
│   └── datasets/              # Original UCI HAR dataset
│
├── docs/                       # Documentation
│   ├── llm_usage.md           # LLM usage log
│   ├── methodology.md         # Technical methodology
│   ├── SETUP_GUIDE.md         # This file
│   └── presentation_slides.md # Presentation content
│
├── results/                    # Training results
│   ├── training_results.png   # Training curves
│   └── post_synthetic_analysis/ # Data quality reports
│
├── screenshots/                # Application screenshots
│   └── api/                   # API screenshots
│
└── tests/                      # Test suite
    ├── test_api.py            # API tests
    └── test_real_world.py     # Real-world data tests
```

---

## Next Steps

### For Users
1. Run Streamlit app: `streamlit run app.py`
2. Try demo mode to see how it works
3. Collect your own data with Physics Toolbox
4. Test authentication with real-world data

### For Developers
1. Explore Jupyter notebooks
2. Review source code in `src/`
3. Train custom models
4. Extend functionality

### For Researchers
1. Read `docs/methodology.md` for technical details
2. Review `docs/llm_usage.md` for LLM integration
3. Analyze results in `results/` directory
4. Experiment with different architectures

---

## Additional Resources

### Documentation
- **README.md**: Project overview and quick start
- **docs/methodology.md**: Technical methodology
- **docs/llm_usage.md**: LLM usage documentation
- **PRESENTATION.md**: 7-slide presentation

### External Links
- **UCI HAR Dataset**: https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
- **Physics Toolbox**: https://play.google.com/store/apps/details?id=com.chrystianvieyra.physicstoolboxsuite
- **Streamlit Docs**: https://docs.streamlit.io/
- **PyTorch Docs**: https://pytorch.org/docs/

---

## Support

### Getting Help
- **Email**: admin@docu3c.com
- **GitHub Issues**: [Repository Issues Page]
- **Documentation**: See `docs/` directory

### Reporting Bugs
1. Check existing issues
2. Provide error message
3. Include system information
4. Describe steps to reproduce

### Contributing
1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

---

## License

This project is developed for the Docu3C AI Challenge.

---

## Acknowledgments

- UCI Machine Learning Repository for HAR dataset
- Physics Toolbox Sensor Suite for data collection
- Claude 3.5 Sonnet for LLM assistance
- Streamlit for web framework

---

**Setup Complete!** You're ready to use the gait-based authentication system.

For questions or issues, refer to the troubleshooting section or contact admin@docu3c.com
