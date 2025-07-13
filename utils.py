#!/usr/bin/env python3
"""
Utility functions for Iris Dataset Classification
===============================================

This module contains utility functions for data visualization,
model evaluation, and result analysis.

Author: Anjali Gopi
GitHub: @anjaligopi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc

def create_sequence_diagram():
    """
    Create a sequence diagram showing the ML pipeline flow.
    This is a text-based representation for the README.
    """
    sequence_diagram = """
    ```mermaid
    sequenceDiagram
        participant U as User
        participant D as DataLoader
        participant P as Preprocessor
        participant M as Models
        participant E as Evaluator
        participant V as Visualizer
        
        U->>D: Load Iris Dataset
        D->>P: Raw Data
        P->>P: Split Train/Test
        P->>P: Feature Scaling
        P->>M: Preprocessed Data
        
        par Model Training
            M->>M: Logistic Regression
            M->>M: SVM
            M->>M: Random Forest
            M->>M: Decision Tree
            M->>M: KNN
            M->>M: Naive Bayes
        end
        
        M->>E: Trained Models
        E->>E: Calculate Metrics
        E->>V: Results
        
        V->>V: Generate Plots
        V->>U: Final Report
    ```
    """
    return sequence_diagram

def create_flowchart():
    """
    Create a flowchart showing the project execution flow.
    """
    flowchart = """
    ```mermaid
    flowchart TD
        A[Start] --> B[Load Iris Dataset]
        B --> C[Data Exploration]
        C --> D[Data Preprocessing]
        D --> E[Split Train/Test]
        E --> F[Feature Scaling]
        F --> G[Initialize Models]
        G --> H[Train Models]
        H --> I[Make Predictions]
        I --> J[Calculate Metrics]
        J --> K[Generate Visualizations]
        K --> L[Cross-Validation]
        L --> M[Hyperparameter Tuning]
        M --> N[Save Results]
        N --> O[End]
        
        style A fill:#e1f5fe
        style O fill:#e8f5e8
        style J fill:#fff3e0
        style K fill:#fff3e0
    ```
    """
    return flowchart

def plot_model_comparison(results, save_path='model_comparison.png'):
    """
    Create a comprehensive model comparison plot.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    save_path : str
        Path to save the plot
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        values = [results[name][metric] for name in results.keys()]
        names = list(results.keys())
        
        bars = axes[i].bar(names, values, color=sns.color_palette("husl", len(names)))
        axes[i].set_title(f'{metric.title()} Comparison')
        axes[i].set_ylabel(metric.title())
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_interactive_plots(results, target_names, feature_names):
    """
    Create interactive plots using Plotly.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    target_names : list
        List of target class names
    feature_names : list
        List of feature names
    """
    # Model comparison bar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[metric.title() for metric in metrics],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    for i, metric in enumerate(metrics):
        row = i // 2 + 1
        col = i % 2 + 1
        
        values = [results[name][metric] for name in results.keys()]
        names = list(results.keys())
        
        fig.add_trace(
            go.Bar(x=names, y=values, name=metric.title()),
            row=row, col=col
        )
    
    fig.update_layout(
        title="Model Performance Comparison",
        height=800,
        showlegend=False
    )
    
    fig.write_html("interactive_model_comparison.html")
    
    # ROC curves
    fig_roc = go.Figure()
    
    for name, result in results.items():
        for i in range(len(target_names)):
            y_true_binary = (result['y_test'] == i).astype(int)
            y_score = result['y_pred_proba'][:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    name=f'{name} - {target_names[i]} (AUC = {roc_auc:.3f})',
                    mode='lines'
                )
            )
    
    fig_roc.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], name='Random',
                  line=dict(dash='dash', color='black'))
    )
    
    fig_roc.update_layout(
        title="ROC Curves (One-vs-Rest)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=800, height=600
    )
    
    fig_roc.write_html("interactive_roc_curves.html")

def generate_model_report(results, target_names):
    """
    Generate a comprehensive model report.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    target_names : list
        List of target class names
    
    Returns:
    --------
    str
        Formatted report
    """
    report = "IRIS CLASSIFICATION MODEL REPORT\n"
    report += "="*50 + "\n\n"
    
    # Overall summary
    report += "OVERALL SUMMARY:\n"
    report += "-" * 20 + "\n"
    
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    report += f"Best Model: {best_model}\n"
    report += f"Best Accuracy: {results[best_model]['accuracy']:.4f}\n\n"
    
    # Detailed results for each model
    for name, result in results.items():
        report += f"{name.upper()}:\n"
        report += "-" * len(name) + "\n"
        report += f"Accuracy: {result['accuracy']:.4f}\n"
        report += f"Precision: {result['precision']:.4f}\n"
        report += f"Recall: {result['recall']:.4f}\n"
        report += f"F1-Score: {result['f1_score']:.4f}\n\n"
        
        # Confusion matrix
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        report += "Confusion Matrix:\n"
        for i, row in enumerate(cm):
            report += f"  {target_names[i]}: {row}\n"
        report += "\n"
    
    return report

def save_model_results(results, target_names, output_file='model_report.txt'):
    """
    Save model results to a file.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    target_names : list
        List of target class names
    output_file : str
        Output file path
    """
    report = generate_model_report(results, target_names)
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Model report saved to {output_file}")

def create_feature_importance_plot(model, feature_names, model_name, save_path=None):
    """
    Create feature importance plot for tree-based models.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(importances)), importances[indices])
        plt.title(f'{model_name}\nFeature Importance')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(range(len(importances)), 
                   [feature_names[i] for i in indices], rotation=45)
        
        # Add value labels on bars
        for bar, importance in zip(bars, importances[indices]):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{importance:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    else:
        print(f"{model_name} does not support feature importance analysis.")

def validate_input_data(sepal_length, sepal_width, petal_length, petal_width):
    """
    Validate input data for iris species prediction.
    
    Parameters:
    -----------
    sepal_length : float
        Sepal length in cm
    sepal_width : float
        Sepal width in cm
    petal_length : float
        Petal length in cm
    petal_width : float
        Petal width in cm
    
    Returns:
    --------
    bool
        True if data is valid, False otherwise
    """
    # Check for reasonable ranges based on iris dataset
    ranges = {
        'sepal_length': (4.0, 8.0),
        'sepal_width': (2.0, 4.5),
        'petal_length': (1.0, 7.0),
        'petal_width': (0.1, 2.5)
    }
    
    values = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    
    for feature, value in values.items():
        min_val, max_val = ranges[feature]
        if not (min_val <= value <= max_val):
            print(f"Warning: {feature} value {value} is outside expected range [{min_val}, {max_val}]")
            return False
    
    return True

def create_prediction_pipeline(model, scaler=None, target_names=None):
    """
    Create a prediction pipeline function.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    scaler : sklearn scaler, optional
        Fitted scaler for feature scaling
    target_names : list, optional
        List of target class names
    
    Returns:
    --------
    function
        Prediction function
    """
    def predict_species(sepal_length, sepal_width, petal_length, petal_width):
        """
        Predict iris species based on measurements.
        
        Parameters:
        -----------
        sepal_length : float
            Sepal length in cm
        sepal_width : float
            Sepal width in cm
        petal_length : float
            Petal length in cm
        petal_width : float
            Petal width in cm
        
        Returns:
        --------
        tuple
            (predicted_species, confidence, is_valid)
        """
        # Validate input
        is_valid = validate_input_data(sepal_length, sepal_width, petal_length, petal_width)
        
        if not is_valid:
            return None, 0.0, False
        
        # Create feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale features if necessary
        if scaler is not None:
            features = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        confidence = probability[prediction]
        
        # Get species name
        if target_names is not None:
            species = target_names[prediction]
        else:
            species = f"Class_{prediction}"
        
        return species, confidence, True
    
    return predict_species 