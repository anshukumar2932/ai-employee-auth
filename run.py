#!/usr/bin/env python3
"""
Quick run script for the Gait Authentication System
"""
import argparse
import subprocess
import sys
import os

def run_training():
    """Run the training notebook"""
    print(" Starting training...")
    os.chdir('notebooks')
    subprocess.run(['jupyter', 'notebook', 'train.ipynb'])

def run_api():
    """Start the API server"""
    print(" Starting API server...")
    subprocess.run([sys.executable, 'src/api.py'])

def run_tests():
    """Run the test suite"""
    print(" Running tests...")
    subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'])

def create_demo():
    """Create demo data"""
    print("Creating demo data...")
    subprocess.run([sys.executable, 'src/real_world_test.py', '--create_demo'])

def main():
    parser = argparse.ArgumentParser(description='Gait Authentication System Runner')
    parser.add_argument('command', choices=['train', 'api', 'test', 'demo'], 
                       help='Command to run')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_training()
    elif args.command == 'api':
        run_api()
    elif args.command == 'test':
        run_tests()
    elif args.command == 'demo':
        create_demo()

if __name__ == '__main__':
    main()