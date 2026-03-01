"""
Gait Identification Model Training
Implements multiple models from simple to complex for comparison
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class GaitDataset(Dataset):
    """PyTorch Dataset for raw signal data"""
    def __init__(self, body_acc_x, body_acc_y, body_acc_z, subjects):
        self.signals = np.stack([body_acc_x, body_acc_y, body_acc_z], axis=2)
        self.labels = subjects - 1  # Convert to 0-indexed
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.signals[idx]),
            torch.LongTensor([self.labels[idx]])[0]
        )


class SimpleCNN(nn.Module):
    """Simple 1D CNN for gait identification"""
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 128, 3)
        x = x.permute(0, 2, 1)  # -> (batch, 3, 128)
        x = self.conv(x)
        x = x.squeeze(-1)
        return self.fc(x)


class GaitTrainer:
    """Main training class for all models"""
    
    def __init__(self, data_path='data/cleaned_walking_data'):
        self.data_path = Path(data_path)
        self.results = {}
        self.load_data()
        
    def load_data(self):
        """Load cleaned walking data"""
        print("Loading data...")
        
        # Load training data
        train_path = self.data_path / 'train'
        self.X_train_signals = np.stack([
            np.load(train_path / 'body_acc_x.npy'),
            np.load(train_path / 'body_acc_y.npy'),
            np.load(train_path / 'body_acc_z.npy')
        ], axis=-1)
        self.X_train_features = np.load(train_path / 'features.npy')
        self.y_train = np.load(train_path / 'subjects.npy')
        
        # Load test data
        test_path = self.data_path / 'test'
        self.X_test_signals = np.stack([
            np.load(test_path / 'body_acc_x.npy'),
            np.load(test_path / 'body_acc_y.npy'),
            np.load(test_path / 'body_acc_z.npy')
        ], axis=-1)
        self.X_test_features = np.load(test_path / 'features.npy')
        self.y_test = np.load(test_path / 'subjects.npy')
        
        # Get number of unique subjects
        self.num_classes = len(np.unique(np.concatenate([self.y_train, self.y_test])))
        
        print(f"Training samples: {len(self.y_train)}")
        print(f"Test samples: {len(self.y_test)}")
        print(f"Number of subjects: {self.num_classes}")
        print(f"Signal shape: {self.X_train_signals.shape}")
        print(f"Features shape: {self.X_train_features.shape}")
        
    def train_logistic_regression(self):
        """1. Logistic Regression - Fastest Baseline"""
        print("\n" + "="*60)
        print("Training Logistic Regression (Baseline)")
        print("="*60)
        
        model = LogisticRegression(
            max_iter=2000,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(self.X_train_features, self.y_train)
        y_pred = model.predict(self.X_test_features)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        self.results['Logistic Regression'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'model': model
        }
        
        return model, accuracy
    
    def train_random_forest(self):
        """2. Random Forest - Best Simple Baseline"""
        print("\n" + "="*60)
        print("Training Random Forest (Recommended Baseline)")
        print("="*60)
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(self.X_train_features, self.y_train)
        y_pred = model.predict(self.X_test_features)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Feature importance
        feature_importance = model.feature_importances_
        top_features = np.argsort(feature_importance)[-10:]
        print(f"\nTop 10 most important features: {top_features}")
        
        self.results['Random Forest'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'model': model,
            'feature_importance': feature_importance
        }
        
        return model, accuracy
    
    def train_svm(self):
        """3. SVM - Very Strong for Small Datasets"""
        print("\n" + "="*60)
        print("Training SVM with RBF Kernel (Best for Small Data)")
        print("="*60)
        
        model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                random_state=42,
                verbose=True
            )
        )
        
        model.fit(self.X_train_features, self.y_train)
        y_pred = model.predict(self.X_test_features)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        self.results['SVM'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'model': model
        }
        
        return model, accuracy
    
    def train_simple_cnn(self, epochs=50, batch_size=32, lr=0.001):
        """4. Simple 1D CNN - Best Simple Deep Model"""
        print("\n" + "="*60)
        print("Training Simple 1D CNN (Deep Learning)")
        print("="*60)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create datasets
        train_dataset = GaitDataset(
            np.load(self.data_path / 'train' / 'body_acc_x.npy'),
            np.load(self.data_path / 'train' / 'body_acc_y.npy'),
            np.load(self.data_path / 'train' / 'body_acc_z.npy'),
            self.y_train
        )
        
        test_dataset = GaitDataset(
            np.load(self.data_path / 'test' / 'body_acc_x.npy'),
            np.load(self.data_path / 'test' / 'body_acc_y.npy'),
            np.load(self.data_path / 'test' / 'body_acc_z.npy'),
            self.y_test
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = SimpleCNN(self.num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        best_acc = 0
        train_losses = []
        train_accs = []
        test_accs = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for signals, labels in train_loader:
                signals, labels = signals.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(signals)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * correct / total
            train_losses.append(total_loss / len(train_loader))
            train_accs.append(train_acc)
            
            # Evaluate
            model.eval()
            correct = 0
            total = 0
            all_preds = []
            
            with torch.no_grad():
                for signals, labels in test_loader:
                    signals, labels = signals.to(device), labels.to(device)
                    outputs = model(signals)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    all_preds.extend(predicted.cpu().numpy())
            
            test_acc = 100. * correct / total
            test_accs.append(test_acc)
            
            scheduler.step(test_acc)
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_preds = np.array(all_preds) + 1  # Convert back to 1-indexed
                torch.save(model.state_dict(), 'models/simple_cnn_best.pth')
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_losses[-1]:.4f}, "
                      f"Train Acc: {train_acc:.2f}%, "
                      f"Test Acc: {test_acc:.2f}%")
        
        accuracy = best_acc / 100
        f1 = f1_score(self.y_test, best_preds, average='weighted')
        
        print(f"\nBest Test Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        self.results['Simple CNN'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': best_preds,
            'model': model,
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_accs': test_accs
        }
        
        return model, accuracy
    
    def plot_confusion_matrix(self, model_name):
        """Plot confusion matrix for a specific model"""
        if model_name not in self.results:
            print(f"Model {model_name} not found in results")
            return
        
        y_pred = self.results[model_name]['predictions']
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Subject')
        plt.xlabel('Predicted Subject')
        plt.tight_layout()
        plt.savefig(f'results/{model_name.replace(" ", "_")}_confusion_matrix.png', dpi=150)
        plt.close()
        print(f"Confusion matrix saved for {model_name}")
    
    def plot_training_curves(self):
        """Plot training curves for CNN"""
        if 'Simple CNN' not in self.results:
            return
        
        result = self.results['Simple CNN']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curve
        ax1.plot(result['train_losses'], label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(result['train_accs'], label='Train Accuracy')
        ax2.plot(result['test_accs'], label='Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Test Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/cnn_training_curves.png', dpi=150)
        plt.close()
        print("Training curves saved")
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'F1 Score': f"{result['f1_score']:.4f}"
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Accuracy', ascending=False)
        print(df.to_string(index=False))
        
        # Save comparison
        df.to_csv('results/model_comparison.csv', index=False)
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        models = [d['Model'] for d in comparison_data]
        accuracies = [float(d['Accuracy']) for d in comparison_data]
        
        bars = plt.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.ylabel('Accuracy')
        plt.title('Model Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=150)
        plt.close()
        print("\nComparison plot saved to results/model_comparison.png")
        
        return df
    
    def save_best_model(self):
        """Save the best performing model"""
        best_model_name = max(self.results.items(), 
                             key=lambda x: x[1]['accuracy'])[0]
        best_result = self.results[best_model_name]
        
        print(f"\nBest Model: {best_model_name}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        
        # Save model metadata
        metadata = {
            'best_model': best_model_name,
            'accuracy': float(best_result['accuracy']),
            'f1_score': float(best_result['f1_score']),
            'num_classes': int(self.num_classes),
            'train_samples': int(len(self.y_train)),
            'test_samples': int(len(self.y_test))
        }
        
        with open('models/best_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Best model metadata saved to models/best_model_metadata.json")
    
    def generate_classification_report(self, model_name):
        """Generate detailed classification report"""
        if model_name not in self.results:
            return
        
        y_pred = self.results[model_name]['predictions']
        report = classification_report(self.y_test, y_pred)
        
        print(f"\n{model_name} - Classification Report:")
        print(report)
        
        # Save report
        with open(f'results/{model_name.replace(" ", "_")}_report.txt', 'w') as f:
            f.write(f"{model_name} - Classification Report\n")
            f.write("="*60 + "\n")
            f.write(report)


def main():
    """Main training pipeline"""
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    # Initialize trainer
    trainer = GaitTrainer()
    
    # Train all models
    print("\nStarting Model Training Pipeline")
    print("="*60)
    
    # 1. Logistic Regression (Baseline)
    trainer.train_logistic_regression()
    
    # 2. Random Forest (Best Simple Model)
    trainer.train_random_forest()
    
    # 3. SVM (Best for Small Data)
    trainer.train_svm()
    
    # 4. Simple CNN (Deep Learning)
    trainer.train_simple_cnn(epochs=50, batch_size=32, lr=0.001)
    
    # Generate reports and visualizations
    print("\nGenerating Reports and Visualizations")
    print("="*60)
    
    for model_name in trainer.results.keys():
        trainer.plot_confusion_matrix(model_name)
        trainer.generate_classification_report(model_name)
    
    trainer.plot_training_curves()
    
    # Compare all models
    comparison_df = trainer.compare_models()
    
    # Save best model
    trainer.save_best_model()
    
    print("\nTraining Complete!")
    print("="*60)
    print("Results saved to:")
    print("  - results/model_comparison.csv")
    print("  - results/model_comparison.png")
    print("  - results/*_confusion_matrix.png")
    print("  - models/best_model_metadata.json")
    print("  - models/simple_cnn_best.pth")


if __name__ == '__main__':
    main()
