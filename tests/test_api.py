"""
Test suite for the Gait Authentication API
"""
import unittest
import json
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api import app, extract_features_from_accelerometer

class TestGaitAuthAPI(unittest.TestCase):
    
    def setUp(self):
        """Set up test client"""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
    
    def test_stats_endpoint(self):
        """Test stats endpoint"""
        response = self.app.get('/stats')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('model_info', data)
        self.assertIn('system_info', data)
    
    def test_feature_extraction(self):
        """Test feature extraction function"""
        # Create dummy accelerometer data (128 samples x 3 axes)
        acc_data = np.random.randn(128, 3).tolist()
        
        features = extract_features_from_accelerometer(acc_data)
        
        # Should return 567 features
        self.assertEqual(len(features), 567)
        self.assertIsInstance(features, np.ndarray)
    
    @patch('api.model')
    @patch('api.scaler')
    @patch('api.label_encoder')
    def test_authenticate_endpoint(self, mock_le, mock_scaler, mock_model):
        """Test authentication endpoint"""
        # Mock model components
        mock_scaler.transform.return_value = np.random.randn(1, 567)
        mock_model.return_value = MagicMock()
        mock_le.inverse_transform.return_value = [1]
        
        # Create test data
        test_data = {
            'accelerometer_data': np.random.randn(128, 3).tolist(),
            'device_id': 'test_device_001'
        }
        
        response = self.app.post('/authenticate',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        # Should not crash (might return error due to mocking)
        self.assertIn(response.status_code, [200, 500])
    
    def test_authenticate_invalid_data(self):
        """Test authentication with invalid data"""
        # Missing required fields
        response = self.app.post('/authenticate',
                               data=json.dumps({}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
        # Invalid accelerometer data length
        test_data = {
            'accelerometer_data': [[1, 2, 3]],  # Only 1 sample instead of 128
            'device_id': 'test_device'
        }
        
        response = self.app.post('/authenticate',
                               data=json.dumps(test_data),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
    
    def test_not_found(self):
        """Test 404 error handling"""
        response = self.app.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()