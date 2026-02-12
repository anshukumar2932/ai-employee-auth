#!/usr/bin/env python3
"""
Mobile Data Processor for Physics Toolbox Integration
Handles real-time accelerometer data processing
"""

import numpy as np
import pandas as pd
from collections import deque
import time
import requests
import json

class MobileGaitProcessor:
    """
    Real-time gait data processor for mobile integration
    """
    
    def __init__(self, api_url="http://localhost:5000", window_size=128, sampling_rate=50):
        self.api_url = api_url
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.buffer = deque(maxlen=window_size * 2)  # 2x window for overlap
        self.last_prediction = None
        self.prediction_history = deque(maxlen=10)
        
    def add_sample(self, x, y, z, timestamp=None):
        """
        Add a single accelerometer sample to the buffer
        
        Args:
            x, y, z: Accelerometer readings in m/s²
            timestamp: Sample timestamp (optional)
        """
        if timestamp is None:
            timestamp = time.time()
            
        sample = {
            'x': x,
            'y': y, 
            'z': z,
            'timestamp': timestamp
        }
        
        self.buffer.append(sample)
        
        # Check if we have enough samples for prediction
        if len(self.buffer) >= self.window_size:
            return self._try_prediction()
        
        return None
    
    def add_batch(self, accelerometer_data):
        """
        Add batch of accelerometer samples
        
        Args:
            accelerometer_data: List of [x, y, z] samples
        """
        results = []
        
        for sample in accelerometer_data:
            if len(sample) >= 3:
                result = self.add_sample(sample[0], sample[1], sample[2])
                if result:
                    results.append(result)
        
        return results
    
    def _try_prediction(self):
        """Try to make a prediction if we have enough data"""
        if len(self.buffer) < self.window_size:
            return None
            
        # Extract latest window
        window_data = list(self.buffer)[-self.window_size:]
        accelerometer_array = [[s['x'], s['y'], s['z']] for s in window_data]
        
        # Make API call
        try:
            result = self._call_api(accelerometer_array)
            
            if result:
                self.last_prediction = result
                self.prediction_history.append(result)
                
            return result
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def _call_api(self, accelerometer_data):
        """Call the authentication API"""
        payload = {
            "accelerometer_data": accelerometer_data,
            "device_id": "mobile_processor",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        response = requests.post(
            f"{self.api_url}/authenticate",
            json=payload,
            timeout=5.0
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return None
    
    def get_stable_prediction(self, min_windows=3, confidence_threshold=0.85):
        """
        Get stable prediction based on recent history
        
        Args:
            min_windows: Minimum number of windows for stable prediction
            confidence_threshold: Minimum confidence for acceptance
        
        Returns:
            Stable prediction result or None
        """
        if len(self.prediction_history) < min_windows:
            return None
        
        # Get recent predictions
        recent = list(self.prediction_history)[-min_windows:]
        
        # Filter by confidence
        high_conf = [p for p in recent if p.get('confidence', 0) >= confidence_threshold]
        
        if len(high_conf) < min_windows // 2:
            return None
        
        # Find most common person ID
        person_ids = [p['person_id'] for p in high_conf]
        person_counts = {}
        
        for pid in person_ids:
            person_counts[pid] = person_counts.get(pid, 0) + 1
        
        # Most frequent person
        best_person = max(person_counts.items(), key=lambda x: x[1])
        
        # Calculate average confidence for this person
        person_predictions = [p for p in high_conf if p['person_id'] == best_person[0]]
        avg_confidence = np.mean([p['confidence'] for p in person_predictions])
        
        return {
            'person_id': best_person[0],
            'confidence': avg_confidence,
            'supporting_windows': best_person[1],
            'total_windows': len(recent),
            'access_granted': avg_confidence >= confidence_threshold
        }
    
    def process_csv_file(self, csv_path):
        """
        Process entire CSV file and return all predictions
        
        Args:
            csv_path: Path to Physics Toolbox CSV file
        
        Returns:
            List of prediction results
        """
        print(f"Processing CSV file: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Handle different CSV formats
        if 'accelerometer(X)' in df.columns:
            acc_data = df[['accelerometer(X)', 'accelerometer(Y)', 'accelerometer(Z)']].values
        elif 'ax' in df.columns:
            acc_data = df[['ax', 'ay', 'az']].values
        else:
            acc_data = df.iloc[:, 1:4].values  # Assume first column is time
        
        print(f"Loaded {len(acc_data)} samples")
        
        # Process in batches
        results = []
        batch_size = self.window_size // 2  # 50% overlap
        
        for i in range(0, len(acc_data) - self.window_size, batch_size):
            window = acc_data[i:i + self.window_size]
            
            # Clear buffer and add window
            self.buffer.clear()
            for sample in window:
                self.buffer.append({
                    'x': sample[0],
                    'y': sample[1], 
                    'z': sample[2],
                    'timestamp': time.time()
                })
            
            # Get prediction
            result = self._try_prediction()
            if result:
                result['window_start'] = i
                result['window_end'] = i + self.window_size
                results.append(result)
        
        return results
    
    def get_statistics(self):
        """Get processor statistics"""
        return {
            'buffer_size': len(self.buffer),
            'max_buffer_size': self.buffer.maxlen,
            'prediction_history_size': len(self.prediction_history),
            'last_prediction': self.last_prediction,
            'window_size': self.window_size,
            'sampling_rate': self.sampling_rate
        }

# Example usage and testing
def main():
    """Example usage of MobileGaitProcessor"""
    
    print("Mobile Gait Processor Demo")
    print("=" * 50)
    
    # Initialize processor
    processor = MobileGaitProcessor()
    
    # Test with synthetic data
    print("\n1. Testing with synthetic walking data...")
    
    # Generate synthetic walking pattern
    t = np.linspace(0, 5, 250)  # 5 seconds at 50Hz
    freq = 2.0  # 2 Hz walking frequency
    
    acc_x = 0.5 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
    acc_y = 0.3 * np.cos(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
    acc_z = 9.8 + 0.2 * np.sin(4 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
    
    # Process samples one by one (simulating real-time)
    predictions = []
    
    for i, (x, y, z) in enumerate(zip(acc_x, acc_y, acc_z)):
        result = processor.add_sample(x, y, z)
        
        if result:
            print(f"   Sample {i}: Person {result['person_id']}, "
                  f"Confidence: {result['confidence']:.2%}")
            predictions.append(result)
    
    print(f"\nGenerated {len(predictions)} predictions")
    
    # Get stable prediction
    stable = processor.get_stable_prediction()
    if stable:
        print(f"\nStable Prediction:")
        print(f"   Person: {stable['person_id']}")
        print(f"   Confidence: {stable['confidence']:.2%}")
        print(f"   Supporting windows: {stable['supporting_windows']}/{stable['total_windows']}")
        print(f"   Access: {'Granted' if stable['access_granted'] else 'Denied'}")
    
    # Show statistics
    stats = processor.get_statistics()
    print(f"\nProcessor Statistics:")
    for key, value in stats.items():
        if key != 'last_prediction':
            print(f"   {key}: {value}")

if __name__ == "__main__":
    main()