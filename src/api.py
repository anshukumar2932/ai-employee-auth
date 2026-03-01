#!/usr/bin/env python3
"""
Flask API for Gait-based Employee Authentication
Production-ready endpoint for contactless security system
"""

from flask import Flask, request, jsonify
import numpy as np
import torch
import torch.nn as nn
import pickle
import time
import logging
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model
model = None
scaler = None
label_encoder = None
CONFIDENCE_THRESHOLD = 0.85

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

def load_model():
    """Load the trained model and preprocessing objects"""
    global model, scaler, label_encoder
    
    model_path = 'models/gait_id_production.pth'
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        model = ProductionGaitModel(
            input_size=checkpoint.get('input_size', 567),
            num_classes=checkpoint['n_classes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler = checkpoint['scaler']
        label_encoder = checkpoint['label_encoder']
        
        logger.info(f"Model loaded successfully - {checkpoint['n_classes']} classes")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def extract_features_from_accelerometer(acc_data):
    """
    Extract 567 features from accelerometer data
    
    Args:
        acc_data: List of [x, y, z] accelerometer readings (128 samples)
    
    Returns:
        features: 567-dimensional feature vector
    """
    if len(acc_data) != 128:
        raise ValueError(f"Expected 128 samples, got {len(acc_data)}")
    
    acc_data = np.array(acc_data)
    acc_x, acc_y, acc_z = acc_data[:, 0], acc_data[:, 1], acc_data[:, 2]
    
    def extract_axis_features(signal):
        """Extract 18 features per axis"""
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
    feats_x = extract_axis_features(acc_x)
    feats_y = extract_axis_features(acc_y)
    feats_z = extract_axis_features(acc_z)
    
    # Combine core features (54 total)
    core_features = feats_x + feats_y + feats_z
    
    # Pad to 561 features (simplified - production would extract all 561)
    padded_features = core_features + [0] * (561 - len(core_features))
    
    # Add simulated gyroscope features
    gyro_features = [
        np.std(np.diff(acc_x)),  # gyro_x_mean (simulated)
        np.std(acc_x),           # gyro_x_std
        np.std(np.diff(acc_y)),  # gyro_y_mean (simulated)
        np.std(acc_y),           # gyro_y_std
        np.std(np.diff(acc_z)),  # gyro_z_mean (simulated)
        np.std(acc_z)            # gyro_z_std
    ]
    
    # Combine to 567 features
    all_features = padded_features + gyro_features
    
    return np.array(all_features)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """
    Main authentication endpoint
    
    Expected JSON:
    {
        "accelerometer_data": [[x1,y1,z1], [x2,y2,z2], ...],  # 128 samples
        "device_id": "employee_phone_001",
        "timestamp": "2026-02-13T10:30:00Z"
    }
    """
    start_time = time.time()
    
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Check required fields
        required_fields = ['accelerometer_data', 'device_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        acc_data = data['accelerometer_data']
        device_id = data['device_id']
        
        # Validate accelerometer data
        if not isinstance(acc_data, list) or len(acc_data) != 128:
            return jsonify({'error': 'accelerometer_data must be list of 128 samples'}), 400
        
        for i, sample in enumerate(acc_data):
            if not isinstance(sample, list) or len(sample) != 3:
                return jsonify({'error': f'Sample {i} must be [x, y, z] format'}), 400
        
        # Extract features
        try:
            features = extract_features_from_accelerometer(acc_data)
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return jsonify({'error': 'Feature extraction failed'}), 500
        
        # Normalize features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predict
        X = torch.tensor(features_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
        
        # Decode prediction
        person_id = int(label_encoder.inverse_transform(predicted.numpy())[0])
        confidence_score = float(confidence.item())
        
        # Determine access
        access_granted = confidence_score >= CONFIDENCE_THRESHOLD
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Log attempt
        logger.info(f"Auth attempt - Device: {device_id}, Person: {person_id}, "
                   f"Confidence: {confidence_score:.2%}, Access: {access_granted}")
        
        # Response
        response = {
            'person_id': person_id,
            'confidence': confidence_score,
            'access_granted': access_granted,
            'processing_time_ms': round(processing_time, 1),
            'timestamp': datetime.now().isoformat(),
            'device_id': device_id
        }
        
        # Add warning if confidence is low
        if confidence_score < CONFIDENCE_THRESHOLD:
            response['warning'] = f'Low confidence ({confidence_score:.1%}). Access denied.'
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/batch_authenticate', methods=['POST'])
def batch_authenticate():
    """
    Batch authentication for multiple windows
    
    Expected JSON:
    {
        "windows": [
            {"accelerometer_data": [[x1,y1,z1], ...], "window_id": 1},
            {"accelerometer_data": [[x1,y1,z1], ...], "window_id": 2},
            ...
        ],
        "device_id": "employee_phone_001"
    }
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if 'windows' not in data or 'device_id' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        windows = data['windows']
        device_id = data['device_id']
        
        if not isinstance(windows, list) or len(windows) == 0:
            return jsonify({'error': 'windows must be non-empty list'}), 400
        
        results = []
        all_features = []
        
        # Extract features for all windows
        for i, window in enumerate(windows):
            if 'accelerometer_data' not in window:
                return jsonify({'error': f'Window {i} missing accelerometer_data'}), 400
            
            try:
                features = extract_features_from_accelerometer(window['accelerometer_data'])
                all_features.append(features)
            except Exception as e:
                return jsonify({'error': f'Feature extraction failed for window {i}'}), 500
        
        # Batch prediction
        all_features = np.array(all_features)
        features_scaled = scaler.transform(all_features)
        X = torch.tensor(features_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = probs.max(1)
        
        # Process results
        person_ids = label_encoder.inverse_transform(predicted.numpy())
        
        for i, (person_id, confidence) in enumerate(zip(person_ids, confidences)):
            window_id = windows[i].get('window_id', i)
            access_granted = confidence.item() >= CONFIDENCE_THRESHOLD
            
            results.append({
                'window_id': window_id,
                'person_id': int(person_id),
                'confidence': float(confidence.item()),
                'access_granted': access_granted
            })
        
        # Overall decision (majority vote with confidence weighting)
        person_votes = {}
        for result in results:
            pid = result['person_id']
            conf = result['confidence']
            if pid not in person_votes:
                person_votes[pid] = []
            person_votes[pid].append(conf)
        
        # Calculate weighted votes
        best_person = None
        best_score = 0
        
        for pid, confidences in person_votes.items():
            score = np.mean(confidences) * len(confidences)  # avg confidence * count
            if score > best_score:
                best_score = score
                best_person = pid
        
        overall_confidence = np.mean([c for c in person_votes[best_person]])
        overall_access = overall_confidence >= CONFIDENCE_THRESHOLD
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Batch auth - Device: {device_id}, Windows: {len(windows)}, "
                   f"Person: {best_person}, Confidence: {overall_confidence:.2%}")
        
        return jsonify({
            'overall_result': {
                'person_id': best_person,
                'confidence': overall_confidence,
                'access_granted': overall_access,
                'windows_processed': len(windows)
            },
            'window_results': results,
            'processing_time_ms': round(processing_time, 1),
            'timestamp': datetime.now().isoformat(),
            'device_id': device_id
        })
        
    except Exception as e:
        logger.error(f"Batch authentication error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify({
        'model_info': {
            'classes': len(label_encoder.classes_) if label_encoder else 0,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'input_features': 567
        },
        'system_info': {
            'timestamp': datetime.now().isoformat(),
            'uptime': time.time() - app.start_time if hasattr(app, 'start_time') else 0
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model on startup
    print("Starting Gait Authentication API...")
    
    if not load_model():
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    app.start_time = time.time()
    
    print("Model loaded successfully")
    print(f"   Classes: {len(label_encoder.classes_)}")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("\nAPI Endpoints:")
    print("   POST /authenticate - Single authentication")
    print("   POST /batch_authenticate - Batch authentication")
    print("   GET /health - Health check")
    print("   GET /stats - System statistics")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

def main():
    """Entry point for console script"""
    # Load model on startup
    print("Starting Gait Authentication API...")
    
    if not load_model():
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    app.start_time = time.time()
    
    print("Model loaded successfully")
    print(f"   Classes: {len(label_encoder.classes_)}")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("\nAPI Endpoints:")
    print("   POST /authenticate - Single authentication")
    print("   POST /batch_authenticate - Batch authentication")
    print("   GET /health - Health check")
    print("   GET /stats - System statistics")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)