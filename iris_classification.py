#!/usr/bin/env python3
"""
Iris Dataset Classification with Traditional ML Algorithms
=======================================================

This script implements multiple traditional machine learning algorithms
for the iris dataset classification problem with comprehensive evaluation metrics.

Author: Anjali Gopi
GitHub: @anjaligopi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')

class IrisClassifier:
    """
    A comprehensive classifier for the iris dataset with multiple algorithms
    and evaluation metrics.
    """
    
    def __init__(self):
        """Initialize the classifier with data loading and preprocessing."""
        # Load the iris dataset
        self.iris = load_iris()
        self.X = self.iris.data
        self.y = self.iris.target
        self.feature_names = self.iris.feature_names
        self.target_names = self.iris.target_names
        
        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Initialize scaler for feature scaling
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Initialize models dictionary
        self.models = {}
        self.results = {}
        
    def initialize_models(self):
        """Initialize all traditional ML models for classification."""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Support Vector Machine': SVC(random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        
    def train_models(self):
        """Train all models and store results."""
        print("Training models...")
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            if name in ['Logistic Regression', 'Support Vector Machine']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            # Store results
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
        print("Training completed!")
        
    def evaluate_models(self):
        """Evaluate all models with comprehensive metrics."""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Create results DataFrame
        results_data = []
        for name, result in self.results.items():
            results_data.append({
                'Model': name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1_score']:.4f}"
            })
        
        results_df = pd.DataFrame(results_data)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, result['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.target_names,
                       yticklabels=self.target_names,
                       ax=axes[idx])
            axes[idx].set_title(f'{name}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_roc_curves(self):
        """Plot ROC curves for all models (one-vs-rest)."""
        plt.figure(figsize=(12, 8))
        
        for name, result in self.results.items():
            # Calculate ROC curve for each class
            for i in range(len(self.target_names)):
                y_true_binary = (self.y_test == i).astype(int)
                y_score = result['y_pred_proba'][:, i]
                
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2,
                        label=f'{name} - {self.target_names[i]} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_importance(self):
        """Plot feature importance for models that support it."""
        models_with_importance = ['Random Forest', 'Decision Tree']
        
        fig, axes = plt.subplots(1, len(models_with_importance), figsize=(15, 5))
        if len(models_with_importance) == 1:
            axes = [axes]
            
        for idx, name in enumerate(models_with_importance):
            if name in self.results:
                model = self.results[name]['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    axes[idx].bar(range(len(importances)), importances[indices])
                    axes[idx].set_title(f'{name}\nFeature Importance')
                    axes[idx].set_xlabel('Features')
                    axes[idx].set_ylabel('Importance')
                    axes[idx].set_xticks(range(len(importances)))
                    axes[idx].set_xticklabels([self.feature_names[i] for i in indices], rotation=45)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def cross_validation_analysis(self):
        """Perform cross-validation analysis for all models."""
        print("\n" + "="*60)
        print("CROSS-VALIDATION RESULTS (5-fold)")
        print("="*60)
        
        cv_results = {}
        for name, model in self.models.items():
            # Use scaled data for models that benefit from it
            if name in ['Logistic Regression', 'Support Vector Machine']:
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            cv_results[name] = {
                'mean_cv_score': cv_scores.mean(),
                'std_cv_score': cv_scores.std(),
                'cv_scores': cv_scores
            }
            
            print(f"{name}:")
            print(f"  Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  CV Scores: {cv_scores}")
            print()
            
        return cv_results
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best performing model."""
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        # Find the best model based on accuracy
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['accuracy'])
        
        print(f"Performing hyperparameter tuning for {best_model_name}...")
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Support Vector Machine': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear']
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if best_model_name in param_grids:
            model = self.models[best_model_name]
            param_grid = param_grids[best_model_name]
            
            # Use appropriate data for tuning
            if best_model_name in ['Logistic Regression', 'Support Vector Machine']:
                X_tune = self.X_train_scaled
            else:
                X_tune = self.X_train
            
            # Perform GridSearchCV
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_tune, self.y_train)
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
            # Update the best model
            self.models[best_model_name] = grid_search.best_estimator_
            
            return grid_search
        else:
            print(f"Hyperparameter tuning not implemented for {best_model_name}")
            return None
    
    def generate_classification_report(self):
        """Generate detailed classification report for all models."""
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORTS")
        print("="*60)
        
        for name, result in self.results.items():
            print(f"\n{name}:")
            print("-" * 40)
            report = classification_report(
                self.y_test, result['y_pred'],
                target_names=self.target_names
            )
            print(report)
    
    def save_results(self):
        """Save all results to files."""
        # Save results DataFrame
        results_df = self.evaluate_models()
        results_df.to_csv('model_results.csv', index=False)
        
        # Save detailed results
        with open('detailed_results.txt', 'w') as f:
            f.write("IRIS DATASET CLASSIFICATION RESULTS\n")
            f.write("="*50 + "\n\n")
            
            for name, result in self.results.items():
                f.write(f"{name}:\n")
                f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  Recall: {result['recall']:.4f}\n")
                f.write(f"  F1-Score: {result['f1_score']:.4f}\n")
                f.write("\n")
        
        print("Results saved to 'model_results.csv' and 'detailed_results.txt'")

def main():
    """Main function to run the complete iris classification analysis."""
    print("IRIS DATASET CLASSIFICATION ANALYSIS")
    print("="*50)
    
    # Initialize classifier
    classifier = IrisClassifier()
    
    # Initialize and train models
    classifier.initialize_models()
    classifier.train_models()
    
    # Evaluate models
    results_df = classifier.evaluate_models()
    
    # Generate plots
    classifier.plot_confusion_matrices()
    classifier.plot_roc_curves()
    classifier.plot_feature_importance()
    
    # Cross-validation analysis
    cv_results = classifier.cross_validation_analysis()
    
    # Hyperparameter tuning
    grid_search = classifier.hyperparameter_tuning()
    
    # Generate detailed reports
    classifier.generate_classification_report()
    
    # Save results
    classifier.save_results()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED!")
    print("="*60)
    print("Generated files:")
    print("- model_results.csv: Summary of all model performances")
    print("- detailed_results.txt: Detailed results for each model")
    print("- confusion_matrices.png: Confusion matrices for all models")
    print("- roc_curves.png: ROC curves for all models")
    print("- feature_importance.png: Feature importance plots")

if __name__ == "__main__":
    main() 