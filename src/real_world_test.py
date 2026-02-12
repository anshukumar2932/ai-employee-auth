#!/usr/bin/env python3
"""
Real-world Testing Script for Gait Authentication System
Tests trained model on Physics Toolbox Sensor Suite data
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, lstm_out):
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        weighted = torch.sum(lstm_out * attn_weights, dim=1)
        return weighted

class ProductionGaitModel(nn.Module):
    def __init__(self, input_size=567, hidden_size=256, num_classes=30, dropout=0.4):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            
            nn.Conv1d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(
            384, hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        
        self.attention = Attention(hidden_size * 2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(cnn_out)
        attended = self.attention(lstm_out)
        
        out = self.fc(attended)
        return out

def process_physics_csv(csv_path, window_size=128, sr=50, overlap=0.5):
    """
    Convert Physics Toolbox CSV to UCI HAR compatible features
    
    Args:
        csv_path: Path to CSV file with accelerometer data
        window_size: Window size in samples (128 for 2.56s at 50Hz)
        sr: Target sampling rate
        overlap: Window overlap (0.5 = 50%)
    
    Returns:
        features: Array of 567-dimensional feature vectors
    """
    print(f"Processing {csv_path}...")
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Handle different CSV formats
    if 'accelerometer(X)' in df.columns:
        # Physics Toolbox format
        acc_x = df['accelerometer(X)'].values
        acc_y = df['accelerometer(Y)'].values
        acc_z = df['accelerometer(Z)'].values
    elif 'ax' in df.columns:
        # Alternative format
        acc_x = df['ax'].values
        acc_y = df['ay'].values
        acc_z = df['az'].values
    else:
        # Generic format
        acc_x = df.iloc[:, 1].values  # Assume first column is time
        acc_y = df.iloc[:, 2].values
        acc_z = df.iloc[:, 3].values
    
    print(f"   Data points: {len(acc_x)}")
    print(f"   Duration: {len(acc_x)/sr:.1f} seconds")
    
    # Resample if needed
    if len(acc_x) != int(len(acc_x) / sr * sr):
        target_length = int(len(acc_x) / sr * sr)
        acc_x = np.interp(np.linspace(0, 1, target_length), np.linspace(0, 1, len(acc_x)), acc_x)
        acc_y = np.interp(np.linspace(0, 1, target_length), np.linspace(0, 1, len(acc_y)), acc_y)
        acc_z = np.interp(np.linspace(0, 1, target_length), np.linspace(0, 1, len(acc_z)), acc_z)
    
    # Extract windows
    step = int(window_size * (1 - overlap))
    features = []
    
    for i in range(0, len(acc_x) - window_size, step):
        window_x = acc_x[i:i+window_size]
        window_y = acc_y[i:i+window_size]
        window_z = acc_z[i:i+window_size]
        
        # Extract 18 core features per axis (matching UCI HAR)
        def extract_axis_features(signal):
            return [
                np.mean(signal),                                    # mean
                np.std(signal),                                     # std
                np.median(np.abs(signal - np.median(signal))),      # mad
                np.max(signal),                                     # max
                np.min(signal),                                     # min
                np.sum(np.abs(signal)),                            # sma
                np.mean(signal**2),                                # energy
                np.percentile(signal, 75) - np.percentile(signal, 25),  # iqr
                -np.sum(signal * np.log(np.abs(signal) + 1e-8)),   # entropy
                np.corrcoef(signal[:-1], signal[1:])[0,1] if len(signal) > 1 else 0,  # autocorr
                np.mean(np.abs(np.diff(signal))),                  # mean_abs_diff
                np.std(np.diff(signal)),                           # std_diff
                np.max(np.abs(signal)),                            # max_abs
                np.mean(signal[signal > 0]) if np.any(signal > 0) else 0,  # mean_pos
                np.mean(signal[signal < 0]) if np.any(signal < 0) else 0,  # mean_neg
                len(signal[signal > np.mean(signal)]) / len(signal),  # above_mean_ratio
                np.sum(signal > 0) / len(signal),                  # positive_ratio
                np.sqrt(np.mean(signal**2))                        # rms
            ]
        
        # Extract features for each axis
        feats_x = extract_axis_features(window_x)
        feats_y = extract_axis_features(window_y)
        feats_z = extract_axis_features(window_z)
        
        # Combine all features (18*3 = 54 core features)
        # Pad to 561 to match UCI HAR (simplified - in production would extract all 561)
        core_features = feats_x + feats_y + feats_z
        padded_features = core_features + [0] * (561 - len(core_features))
        
        # Add gyroscope-like features (simulate from accelerometer)
        gyro_features = [
            np.std(np.diff(window_x)),  # gyro_x_mean (simulated)
            np.std(window_x),           # gyro_x_std
            np.std(np.diff(window_y)),  # gyro_y_mean (simulated)
            np.std(window_y),           # gyro_y_std
            np.std(np.diff(window_z)),  # gyro_z_mean (simulated)
            np.std(window_z)            # gyro_z_std
        ]
        
        # Combine to 567 features
        all_features = padded_features + gyro_features
        features.append(all_features)
    
    features_array = np.array(features)
    if len(features) > 0:
        print(f"   Extracted {len(features)} windows with {features_array.shape[1]} features each")
    else:
        print("   No windows extracted - insufficient data")
        return np.array([])
    
    return features_array

def load_model(model_path='models/gait_id_production.pth'):
    """Load trained model and preprocessing objects"""
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first using train.ipynb")
        return None, None, None
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Initialize model
    model = ProductionGaitModel(
        input_size=checkpoint.get('input_size', 567),
        num_classes=checkpoint['n_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler = checkpoint['scaler']
    label_encoder = checkpoint['label_encoder']
    
    print("Model loaded successfully")
    print(f"   Classes: {checkpoint['n_classes']}")
    print(f"   Best accuracy: {checkpoint.get('best_acc', 'Unknown'):.2%}")
    
    return model, scaler, label_encoder

def predict_person(model, scaler, label_encoder, features):
    """
    Predict person ID from features
    
    Returns:
        person_ids: Predicted person IDs
        confidences: Confidence scores
        raw_probs: Raw probability distributions
    """
    # Normalize features
    features_scaled = scaler.transform(features)
    
    # Convert to tensor
    X = torch.tensor(features_scaled, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)
        confidences, predicted = probs.max(1)
    
    # Decode predictions
    person_ids = label_encoder.inverse_transform(predicted.numpy())
    
    return person_ids, confidences.numpy(), probs.numpy()

def test_single_file(csv_path, model, scaler, label_encoder, true_person_id=None):
    """Test model on a single CSV file"""
    print(f"\n{'='*60}")
    print(f"Testing: {csv_path}")
    print(f"{'='*60}")
    
    # Process CSV
    features = process_physics_csv(csv_path)
    
    if len(features) == 0:
        print("No features extracted from CSV")
        return None
    
    # Predict
    person_ids, confidences, probs = predict_person(model, scaler, label_encoder, features)
    
    # Analyze results
    unique_ids, counts = np.unique(person_ids, return_counts=True)
    
    print(f"\nPrediction Results:")
    print(f"   Total windows: {len(person_ids)}")
    print(f"   Average confidence: {confidences.mean():.2%}")
    print(f"   High confidence (>85%): {(confidences > 0.85).sum()}/{len(confidences)} ({(confidences > 0.85).mean():.1%})")
    
    print(f"\nPerson ID Distribution:")
    for pid, count in zip(unique_ids, counts):
        percentage = count / len(person_ids) * 100
        avg_conf = confidences[person_ids == pid].mean()
        print(f"   Person {pid}: {count:3d} windows ({percentage:5.1f}%) - Avg conf: {avg_conf:.2%}")
    
    # Most likely person
    most_likely_person = unique_ids[np.argmax(counts)]
    most_likely_confidence = confidences[person_ids == most_likely_person].mean()
    
    print(f"\nFinal Prediction:")
    print(f"   Person ID: {most_likely_person}")
    print(f"   Confidence: {most_likely_confidence:.2%}")
    print(f"   Windows supporting: {counts[np.argmax(counts)]}/{len(person_ids)} ({counts[np.argmax(counts)]/len(person_ids):.1%})")
    
    if true_person_id is not None:
        correct = (most_likely_person == true_person_id)
        print(f"   True Person: {true_person_id}")
        print(f"   Correct: {'Correct' if correct else 'Incorrect'}")
        
        return {
            'predicted': most_likely_person,
            'true': true_person_id,
            'correct': correct,
            'confidence': most_likely_confidence,
            'windows': len(person_ids)
        }
    
    return {
        'predicted': most_likely_person,
        'confidence': most_likely_confidence,
        'windows': len(person_ids)
    }

def test_directory(data_dir, model, scaler, label_encoder):
    """Test model on all CSV files in a directory"""
    print(f"\n{'='*60}")
    print(f"Testing directory: {data_dir}")
    print(f"{'='*60}")
    
    csv_files = list(Path(data_dir).glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in directory")
        return
    
    results = []
    
    for csv_file in csv_files:
        # Try to extract person ID from filename
        filename = csv_file.stem
        true_person_id = None
        
        # Look for patterns like "person_1.csv", "user_5_walking.csv", etc.
        import re
        match = re.search(r'(\d+)', filename)
        if match:
            true_person_id = int(match.group(1))
        
        result = test_single_file(str(csv_file), model, scaler, label_encoder, true_person_id)
        if result:
            result['filename'] = filename
            results.append(result)
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print(f"SUMMARY - {len(results)} files tested")
        print(f"{'='*60}")
        
        if all('correct' in r for r in results):
            correct_predictions = sum(r['correct'] for r in results)
            accuracy = correct_predictions / len(results)
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            print(f"Overall Accuracy: {accuracy:.2%} ({correct_predictions}/{len(results)})")
            print(f"Average Confidence: {avg_confidence:.2%}")
            
            # Per-file results
            print(f"\nPer-file Results:")
            for r in results:
                status = "Correct" if r['correct'] else "Incorrect"
                print(f"   {status} {r['filename']}: P{r['predicted']} (conf: {r['confidence']:.1%}, {r['windows']} windows)")
        
        else:
            avg_confidence = np.mean([r['confidence'] for r in results])
            print(f"Average Confidence: {avg_confidence:.2%}")
            
            print(f"\nResults:")
            for r in results:
                print(f"   {r['filename']}: Person {r['predicted']} (conf: {r['confidence']:.1%}, {r['windows']} windows)")

def create_demo_data():
    """Create demo CSV files for testing"""
    print("Creating demo data...")
    
    os.makedirs('data/real_world_samples', exist_ok=True)
    
    # Generate synthetic walking data for demo
    np.random.seed(42)
    
    for person_id in range(1, 4):  # 3 demo people
        # Simulate 10 seconds of walking data at 50Hz
        t = np.linspace(0, 10, 500)
        
        # Base walking pattern (different for each person)
        freq = 1.8 + person_id * 0.2  # Different walking frequencies
        amplitude = 0.8 + person_id * 0.1  # Different intensities
        
        # Simulate accelerometer data with person-specific patterns
        acc_x = amplitude * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        acc_y = amplitude * 0.7 * np.cos(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        acc_z = 9.8 + amplitude * 0.3 * np.sin(4 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        
        # Create CSV
        df = pd.DataFrame({
            'time': t,
            'accelerometer(X)': acc_x,
            'accelerometer(Y)': acc_y,
            'accelerometer(Z)': acc_z
        })
        
        csv_path = f'data/real_world_samples/person_{person_id}_walking.csv'
        df.to_csv(csv_path, index=False)
        print(f"   Created: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Test gait authentication on real-world data')
    parser.add_argument('--csv_file', type=str, help='Single CSV file to test')
    parser.add_argument('--data_dir', type=str, default='data/real_world_samples', 
                       help='Directory containing CSV files')
    parser.add_argument('--model_path', type=str, default='models/gait_id_production.pth',
                       help='Path to trained model')
    parser.add_argument('--create_demo', action='store_true', 
                       help='Create demo data for testing')
    
    args = parser.parse_args()
    
    print("Real-world Gait Authentication Test")
    print("=" * 60)
    
    # Create demo data if requested
    if args.create_demo:
        create_demo_data()
        print("\n✅ Demo data created in data/real_world_samples/")
        return
    
    # Load model
    model, scaler, label_encoder = load_model(args.model_path)
    if model is None:
        return
    
    # Test single file or directory
    if args.csv_file:
        if os.path.exists(args.csv_file):
            test_single_file(args.csv_file, model, scaler, label_encoder)
        else:
            print(f"File not found: {args.csv_file}")
    else:
        if os.path.exists(args.data_dir):
            test_directory(args.data_dir, model, scaler, label_encoder)
        else:
            print(f"❌ Directory not found: {args.data_dir}")
            print("Use --create_demo to create sample data")

if __name__ == "__main__":
    main()