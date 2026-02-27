#!/usr/bin/env python3
"""
Fix the train/test split for gait identification.
For person identification, we need samples from ALL subjects in both train and test.
"""
import numpy as np
from pathlib import Path

# Load original data
DATA_PATH = Path('data/cleaned_walking_data')
train_path = DATA_PATH / 'train'
test_path = DATA_PATH / 'test'

# Load all training data
X_train = np.load(train_path / 'features.npy')
y_train = np.load(train_path / 'subjects.npy')

# Load all test data  
X_test = np.load(test_path / 'features.npy')
y_test = np.load(test_path / 'subjects.npy')

# Combine all data
X_all = np.concatenate([X_train, X_test], axis=0)
y_all = np.concatenate([y_train, y_test], axis=0)

print(f"Total samples: {len(X_all)}")
print(f"Total unique subjects: {len(np.unique(y_all))}")
print(f"Subjects: {sorted(np.unique(y_all))}")

# Create proper train/test split (80/20) with stratification
# Manual stratified split to ensure each subject appears in both train and test
np.random.seed(42)

X_train_new = []
X_test_new = []
y_train_new = []
y_test_new = []

for subject in np.unique(y_all):
    # Get all samples for this subject
    subject_mask = y_all == subject
    X_subject = X_all[subject_mask]
    y_subject = y_all[subject_mask]
    
    # Shuffle
    indices = np.random.permutation(len(X_subject))
    X_subject = X_subject[indices]
    y_subject = y_subject[indices]
    
    # Split 80/20
    split_idx = int(0.8 * len(X_subject))
    
    X_train_new.append(X_subject[:split_idx])
    X_test_new.append(X_subject[split_idx:])
    y_train_new.append(y_subject[:split_idx])
    y_test_new.append(y_subject[split_idx:])

# Concatenate all subjects
X_train_new = np.concatenate(X_train_new, axis=0)
X_test_new = np.concatenate(X_test_new, axis=0)
y_train_new = np.concatenate(y_train_new, axis=0)
y_test_new = np.concatenate(y_test_new, axis=0)

# Shuffle the final datasets
train_indices = np.random.permutation(len(X_train_new))
X_train_new = X_train_new[train_indices]
y_train_new = y_train_new[train_indices]

test_indices = np.random.permutation(len(X_test_new))
X_test_new = X_test_new[test_indices]
y_test_new = y_test_new[test_indices]

print(f"\nNew split:")
print(f"Training samples: {len(X_train_new)}")
print(f"Test samples: {len(X_test_new)}")
print(f"Training subjects: {sorted(np.unique(y_train_new))}")
print(f"Test subjects: {sorted(np.unique(y_test_new))}")

# Verify overlap
train_subjects = set(np.unique(y_train_new))
test_subjects = set(np.unique(y_test_new))
overlap = train_subjects.intersection(test_subjects)
print(f"\nSubjects in both train and test: {len(overlap)} (should be 30)")

# Save the corrected split
np.save(train_path / 'features.npy', X_train_new)
np.save(train_path / 'subjects.npy', y_train_new)
np.save(test_path / 'features.npy', X_test_new)
np.save(test_path / 'subjects.npy', y_test_new)

print("\n✅ Data split fixed and saved!")
print("Now run the notebook again.")
