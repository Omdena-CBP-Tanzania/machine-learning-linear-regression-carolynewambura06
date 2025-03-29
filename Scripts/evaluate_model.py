#!/usr/bin/env python3
# scripts/evaluate_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import os

def load_data():
    """Load datasets and trained model"""
    try:
        # Load model
        model = load('../trained_model.pkl')
        print("Model successfully loaded")
        
        # Load data
        data_dir = r'C:\Users\user\Omdena\machine-learning-linear-regression-carolynewambura06'
        X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze()
        y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).squeeze()
        
        return model, X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None, None, None

def calculate_metrics(y_true, y_pred):
    """Calculate and print evaluation metrics"""
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred)
    }
    
    print("\nModel Performance Metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return metrics

def plot_results(y_test, y_pred):
    """Generate evaluation plots"""
    # Create results directory if not exists
    os.makedirs('../results/figures', exist_ok=True)
    
    # Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Values (MEDV)')
    plt.ylabel('Predicted Values (MEDV)')
    plt.title('Actual vs. Predicted House Prices')
    plt.close()
    
    # Residual Analysis
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True, bins=30)
    plt.title('Residual Distribution')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs. Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    
    plt.tight_layout()
    plt.close()

def compare_feature_sets(feature_sets, X_train, y_train, X_test, y_test):
    """Compare performance across different feature sets"""
    results = {}
    
    for name, features in feature_sets.items():
        try:
            model = LinearRegression().fit(X_train[features], y_train)
            y_pred = model.predict(X_test[features])
            
            results[name] = {
                'Features': features[:3] + ['...'] if len(features) > 3 else features,
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred),
                'R²': r2_score(y_test, y_pred),
                'Num Features': len(features)
            }
        except Exception as e:
            print(f"Error with feature set '{name}': {str(e)}")
            results[name] = {
                'Features': 'ERROR',
                'RMSE': np.nan,
                'MAE': np.nan,
                'R²': np.nan,
                'Num Features': 0
            }
    
    return pd.DataFrame(results).T

def main():
    # Step 1: Load data and model
    model, X_train, X_test, y_train, y_test = load_data()
    if model is None:
        return
    
    # Step 2: Generate predictions
    y_pred = model.predict(X_test)
    print(f"\nData Shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Step 3: Calculate metrics
    metrics = calculate_metrics(y_test, y_pred)
    
    # Step 4: Generate plots
    plot_results(y_test, y_pred)
    print("\nSaved visualization plots to results/figures/")
    
    # Step 5: Feature set comparison
    feature_sets = {
        'Top 3': ['lstat', 'rm', 'ptratio'],
        'Top 5': ['lstat', 'rm', 'ptratio', 'indus', 'tax'],
        'All Features': X_train.columns.tolist()
    }
    
    comparison_results = compare_feature_sets(
        feature_sets=feature_sets,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    print("\nFeature Set Comparison Results:")
    print(comparison_results)
    
    # Save results
    os.makedirs('../results/metrics', exist_ok=True)
    comparison_results.to_csv('../results/metrics/feature_comparison.csv')
    print("\nSaved comparison results to results/metrics/feature_comparison.csv")

if __name__ == "__main__":
    main()