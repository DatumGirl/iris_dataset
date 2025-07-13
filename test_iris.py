#!/usr/bin/env python3
"""
Simple test script for Iris Classification
=========================================

This script demonstrates the basic functionality of the iris classification
project without requiring all dependencies to be installed.

Author: Anjali Gopi
GitHub: @anjaligopi
"""

import sys
import os

def test_imports():
    """Test if required packages can be imported."""
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError:
        print("✗ NumPy not available")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError:
        print("✗ Pandas not available")
        return False
    
    try:
        from sklearn.datasets import load_iris
        print("✓ Scikit-learn imported successfully")
    except ImportError:
        print("✗ Scikit-learn not available")
        return False
    
    return True

def test_iris_dataset():
    """Test loading and basic analysis of iris dataset."""
    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        # Load dataset
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        print(f"✓ Iris dataset loaded successfully")
        print(f"  - Dataset shape: {X.shape}")
        print(f"  - Features: {iris.feature_names}")
        print(f"  - Target classes: {iris.target_names}")
        
        # Basic model training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ Basic model training successful")
        print(f"  - Training set size: {X_train.shape[0]}")
        print(f"  - Test set size: {X_test.shape[0]}")
        print(f"  - Model accuracy: {accuracy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during dataset testing: {e}")
        return False

def test_file_structure():
    """Test if all required files exist."""
    required_files = [
        'README.md',
        'requirements.txt',
        'iris_classification.py',
        'iris_analysis.ipynb',
        'utils.py'
    ]
    
    print("\nChecking project file structure:")
    all_exist = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False
    
    return all_exist

def demonstrate_prediction():
    """Demonstrate a simple prediction example."""
    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        # Load and train a simple model
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Example predictions
        test_cases = [
            (5.1, 3.5, 1.4, 0.2),  # Setosa
            (6.3, 3.3, 4.7, 1.6),  # Versicolor
            (6.3, 3.3, 6.0, 2.5)   # Virginica
        ]
        
        print("\nExample Predictions:")
        print("=" * 30)
        
        for i, (sl, sw, pl, pw) in enumerate(test_cases):
            features = np.array([[sl, sw, pl, pw]])
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            species = iris.target_names[prediction]
            confidence = probability[prediction]
            
            print(f"\nTest Case {i+1}:")
            print(f"  Measurements: Sepal L={sl}, Sepal W={sw}, Petal L={pl}, Petal W={pw}")
            print(f"  Predicted Species: {species}")
            print(f"  Confidence: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during prediction demonstration: {e}")
        return False

def main():
    """Main test function."""
    print("IRIS CLASSIFICATION PROJECT TEST")
    print("=" * 40)
    
    # Test imports
    print("\n1. Testing package imports:")
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n⚠️  Some packages are missing. Install them with:")
        print("   pip install -r requirements.txt")
        return
    
    # Test dataset
    print("\n2. Testing iris dataset:")
    dataset_ok = test_iris_dataset()
    
    # Test file structure
    print("\n3. Testing project structure:")
    structure_ok = test_file_structure()
    
    # Demonstrate prediction
    print("\n4. Demonstrating predictions:")
    prediction_ok = demonstrate_prediction()
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)
    
    if imports_ok and dataset_ok and structure_ok and prediction_ok:
        print("✓ All tests passed!")
        print("✓ Project is ready to use")
        print("\nTo run the full analysis:")
        print("  python iris_classification.py")
        print("\nFor interactive analysis:")
        print("  jupyter notebook iris_analysis.ipynb")
    else:
        print("✗ Some tests failed")
        print("Please check the error messages above")
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    main() 