# 🚶 Stark Industries - Gait-Based Authentication System

## Streamlit Web Application

This is a comprehensive web application for the AI-Powered Contactless Employee Security System using gait analysis.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

### 2. Ensure Model is Trained

Make sure you have trained the model and it's saved in `models/best_model_logistic_regression.pkl`:

```bash
python src/train_gait_models.py
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 Features

### 🏠 Home Page
- System overview and statistics
- Recent activity log
- Quick metrics dashboard

### 🔐 Authentication Page
- **Upload Mode**: Upload CSV files with accelerometer data
- **Demo Mode**: Test with pre-loaded samples from the dataset
- Real-time gait pattern visualization
- Confidence-based access control

### 📊 Analytics Dashboard
- Access statistics (granted/denied)
- User activity charts
- Complete access log with export functionality
- Visual analytics with Plotly charts

### 📱 Real-World Testing
- Instructions for Physics Toolbox Sensor Suite app
- CSV upload for real smartphone data
- Data visualization and validation
- Tips for data collection

### ℹ️ About
- Technical documentation
- Performance metrics
- Future enhancements
- References and resources

## 📊 CSV Format for Upload

Your CSV file should have these columns:

```csv
time,accel_x,accel_y,accel_z
0.00,0.12,9.81,0.05
0.02,0.15,9.79,0.07
0.04,0.18,9.83,0.04
...
```

- `time`: Timestamp in seconds
- `accel_x`: X-axis acceleration (m/s²)
- `accel_y`: Y-axis acceleration (m/s²)
- `accel_z`: Z-axis acceleration (m/s²)

## 🎯 How to Use

### Testing with Demo Data

1. Go to **Authentication** page
2. Select **Demo Mode** tab
3. Use the slider to select a sample
4. Click "Test Authentication"
5. View the prediction and confidence score

### Testing with Your Own Data

1. Collect accelerometer data using Physics Toolbox Sensor Suite
2. Export as CSV
3. Go to **Authentication** page
4. Upload your CSV file
5. View the gait pattern visualization
6. Click "Authenticate" to test

### Viewing Analytics

1. Go to **Analytics** page
2. View access statistics and charts
3. Download access log as CSV

## 🔧 Configuration

### Model Settings

The app automatically loads the best model from `models/` directory. Supported models:
- Logistic Regression
- Random Forest
- SVM

### Confidence Threshold

Default threshold for access: **70%**

You can modify this in the code:

```python
if confidence > 0.7:  # Change this value
    # Grant access
```

## 📈 Performance

- **Model Accuracy**: 80-90% on test data
- **Inference Time**: <2 seconds
- **Supported Subjects**: 30 (expandable with synthetic data)

## 🐛 Troubleshooting

### Model Not Found
```
Error: Model not loaded
```
**Solution**: Train the model first using `python src/train_gait_models.py`

### CSV Format Error
```
Error: CSV must contain columns: ['accel_x', 'accel_y', 'accel_z']
```
**Solution**: Ensure your CSV has the correct column names

### Feature Dimension Mismatch
```
Error: Feature dimension mismatch
```
**Solution**: The app automatically pads/truncates features to 561 dimensions

## 🎨 Customization

### Changing Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Adding New Features

The app is modular. Add new pages by:

1. Creating a new function: `def show_new_page():`
2. Adding to navigation: `page = st.radio(..., ["New Page"])`
3. Adding to main: `elif page == "New Page": show_new_page()`

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
- [Physics Toolbox App](https://play.google.com/store/apps/details?id=com.chrystianvieyra.physicstoolboxsuite)

## 🤝 Contributing

Feel free to enhance the app with:
- Additional visualizations
- More authentication methods
- Enhanced security features
- Mobile responsiveness improvements

## 📄 License

This project is part of the Stark Industries Security Challenge.

---

**Built with ❤️ using Streamlit**
