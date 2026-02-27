"""
Quick Training Example - Minimal Code
Train a Random Forest model in under 5 minutes
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("Quick Gait ID Training - Random Forest")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    DATA_PATH = Path('data/cleaned_walking_data')
    
    X_train = np.load(DATA_PATH / 'train' / 'features.npy')
    y_train = np.load(DATA_PATH / 'train' / 'subjects.npy')
    X_test = np.load(DATA_PATH / 'test' / 'features.npy')
    y_test = np.load(DATA_PATH / 'test' / 'subjects.npy')
    
    print(f"   Training samples: {len(y_train)}")
    print(f"   Test samples: {len(y_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Subjects: {len(np.unique(y_train))}")
    
    # Train model
    print("\n2. Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n3. Evaluating...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Expected: 75-88%")
    
    if accuracy >= 0.80:
        print("✅ Excellent! Model performs very well.")
    elif accuracy >= 0.70:
        print("✅ Good! Model performs well.")
    else:
        print("⚠️  Low accuracy. Try SVM or check data quality.")
    
    # Detailed report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred))
    
    # Save model
    print("\n4. Saving model...")
    import pickle
    Path('models').mkdir(exist_ok=True)
    
    with open('models/quick_rf_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("   Model saved to: models/quick_rf_model.pkl")
    
    # Feature importance
    print("\n5. Top 10 Most Important Features:")
    feature_importance = model.feature_importances_
    top_10_idx = np.argsort(feature_importance)[-10:][::-1]
    
    for i, idx in enumerate(top_10_idx, 1):
        print(f"   {i}. Feature {idx}: {feature_importance[idx]:.4f}")
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  - Test on real data: python src/real_world_test.py")
    print("  - Deploy API: python src/api.py")
    print("  - Try other models: python src/train_gait_models.py")

if __name__ == '__main__':
    main()
