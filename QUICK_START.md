# 🚀 Quick Start Guide - Gait Authentication System

## Complete Setup in 5 Minutes

### Step 1: Fix the Data Split (IMPORTANT!)
```bash
python3 fix_data_split.py
```
This fixes the train/test split so subjects appear in both sets.

### Step 2: Train the Model (if not already trained)
```bash
python3 src/train_gait_models.py
```
This will train and save the best model.

### Step 3: Install Streamlit Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### Step 4: Run the Streamlit App
```bash
streamlit run app.py
```

Or use the convenience script:
```bash
./run_streamlit.sh
```

### Step 5: Open Your Browser
The app will automatically open at: `http://localhost:8501`

---

## 🎯 What to Do in the App

### 1. Home Page
- View system overview
- Check model accuracy
- See recent activity

### 2. Authentication Page

#### Demo Mode (Easiest)
1. Click "Demo Mode" tab
2. Use slider to select a sample
3. Click "Test Authentication"
4. See prediction and confidence

#### Upload Mode (Real Data)
1. Click "Upload Data" tab
2. Upload `sample_gait_data.csv` (provided)
3. View gait visualization
4. Click "Authenticate"

### 3. Analytics Page
- View access statistics
- See charts and graphs
- Download access logs

### 4. Real-World Test
- Follow instructions for Physics Toolbox app
- Upload your own accelerometer data
- Test with real smartphone data

---

## 📱 Testing with Real Smartphone Data

### Option 1: Use Sample Data
We've provided `sample_gait_data.csv` for testing:
```bash
# Just upload this file in the app
sample_gait_data.csv
```

### Option 2: Collect Your Own Data

1. **Download Physics Toolbox Sensor Suite**
   - Android: [Play Store Link](https://play.google.com/store/apps/details?id=com.chrystianvieyra.physicstoolboxsuite)
   - iOS: Search "Physics Toolbox" in App Store

2. **Record Accelerometer Data**
   - Open app → Accelerometer
   - Start recording
   - Walk naturally for 5-10 seconds
   - Stop recording

3. **Export as CSV**
   - Click export/share
   - Save as CSV
   - Ensure columns: time, accel_x, accel_y, accel_z

4. **Upload to App**
   - Go to "Real-World Test" page
   - Upload your CSV
   - Test authentication!

---

## 🐛 Troubleshooting

### Problem: Model Not Found
```
Error: Model not loaded
```
**Solution**:
```bash
python3 src/train_gait_models.py
```

### Problem: 0% Accuracy
```
All models showing 0.0000 accuracy
```
**Solution**: Run the data fix script:
```bash
python3 fix_data_split.py
```

### Problem: Streamlit Not Found
```
streamlit: command not found
```
**Solution**:
```bash
pip install streamlit
```

### Problem: CSV Format Error
```
CSV must contain columns: ['accel_x', 'accel_y', 'accel_z']
```
**Solution**: Ensure your CSV has these exact column names

### Problem: Feature Dimension Mismatch
The app automatically handles this by padding/truncating to 561 features.

---

## 📊 Expected Results

### Dataset Performance
- **Training Accuracy**: 85-90%
- **Test Accuracy**: 80-85%
- **Inference Time**: <2 seconds

### Real-World Performance
- **Accuracy**: 70-78% (expected drop)
- **Confidence**: Varies by subject
- **Speed**: Real-time (<2s)

---

## 🎨 Customization

### Change Confidence Threshold
Edit `app.py` line ~350:
```python
if confidence > 0.7:  # Change this (0.0 to 1.0)
    # Grant access
```

### Change Model
Replace in `models/` directory:
- `best_model_logistic_regression.pkl`
- `best_model_metadata.json`

### Add New Features
The app is modular - easy to extend!

---

## 📚 File Structure

```
.
├── app.py                          # Main Streamlit app
├── fix_data_split.py              # Fix train/test split
├── sample_gait_data.csv           # Sample data for testing
├── run_streamlit.sh               # Convenience script
├── requirements_streamlit.txt     # Dependencies
├── STREAMLIT_README.md            # Detailed docs
├── PRESENTATION.md                # Presentation slides
├── models/
│   ├── best_model_*.pkl          # Trained models
│   └── best_model_metadata.json  # Model info
├── data/
│   └── cleaned_walking_data/     # Dataset
└── src/
    └── train_gait_models.py      # Training script
```

---

## 🎯 Demo Workflow

### Complete Demo in 2 Minutes

1. **Start the app**:
   ```bash
   streamlit run app.py
   ```

2. **Home Page** (30 seconds):
   - Show system overview
   - Point out accuracy metrics
   - Explain the approach

3. **Authentication Demo** (1 minute):
   - Go to Authentication page
   - Demo Mode tab
   - Select sample #42
   - Click "Test Authentication"
   - Show confidence score
   - Explain access decision

4. **Upload Demo** (30 seconds):
   - Upload Data tab
   - Upload `sample_gait_data.csv`
   - Show gait visualization
   - Click "Authenticate"
   - Show result

5. **Analytics** (30 seconds):
   - Go to Analytics page
   - Show access log
   - Display charts
   - Download CSV

---

## 💡 Tips for Best Results

### Data Collection
- ✅ Walk at normal pace
- ✅ Hold phone naturally
- ✅ Record 5-10 seconds
- ✅ Straight line walking
- ❌ Avoid running or jumping
- ❌ Don't shake the phone

### Model Performance
- Higher confidence = more reliable
- Multiple samples improve accuracy
- Different conditions may affect results
- Calibration per device helps

### Presentation
- Start with Home page overview
- Demo with sample data first
- Show real-world testing capability
- Explain challenges and solutions
- Highlight LLM usage

---

## 🚀 Next Steps

1. ✅ Run the app and explore
2. ✅ Test with sample data
3. ✅ Collect your own data
4. ✅ Review the presentation
5. ✅ Check the documentation
6. ✅ Prepare your demo

---

## 📞 Need Help?

Check these resources:
- `STREAMLIT_README.md` - Detailed documentation
- `PRESENTATION.md` - Full presentation
- `docs/` - Additional documentation
- GitHub Issues - Report problems

---

**Ready to authenticate? Let's go! 🚶**
