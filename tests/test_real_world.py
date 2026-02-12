"""
Test suite for real-world data processing
"""
import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from real_world_test import extract_features_from_csv, create_demo_data

class TestRealWorldProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.temp_dir = tempfile.mkdtemp()
    
    def test_create_demo_data(self):
        """Test demo data creation"""
        create_demo_data(self.temp_dir)
        
        # Check if files were created
        files = os.listdir(self.temp_dir)
        self.assertGreater(len(files), 0)
        
        # Check if files are CSV
        csv_files = [f for f in files if f.endswith('.csv')]
        self.assertGreater(len(csv_files), 0)
    
    def test_feature_extraction_from_csv(self):
        """Test feature extraction from CSV"""
        # Create a test CSV file
        test_data = {
            'time': np.linspace(0, 2.56, 128),
            'x': np.random.randn(128),
            'y': np.random.randn(128),
            'z': np.random.randn(128)
        }
        
        df = pd.DataFrame(test_data)
        test_csv = os.path.join(self.temp_dir, 'test.csv')
        df.to_csv(test_csv, index=False)
        
        # Extract features
        features = extract_features_from_csv(test_csv)
        
        # Should return array of features
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[1], 567)  # 567 features per window
    
    def test_invalid_csv(self):
        """Test handling of invalid CSV files"""
        # Create invalid CSV
        invalid_csv = os.path.join(self.temp_dir, 'invalid.csv')
        with open(invalid_csv, 'w') as f:
            f.write("invalid,data\n1,2\n")
        
        # Should handle gracefully
        try:
            features = extract_features_from_csv(invalid_csv)
            # If it doesn't raise an exception, features should be empty or None
            if features is not None:
                self.assertEqual(len(features), 0)
        except Exception:
            # Exception is acceptable for invalid data
            pass
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()