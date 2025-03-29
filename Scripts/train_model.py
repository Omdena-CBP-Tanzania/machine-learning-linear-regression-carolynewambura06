# scripts/train_model.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def load_data():
    """Load training and testing data from CSV files."""
    try:
        X_train = pd.read_csv(r'C:\Users\user\Omdena\machine-learning-linear-regression-carolynewambura06\X_train.csv')
        X_test = pd.read_csv(r'C:\Users\user\Omdena\machine-learning-linear-regression-carolynewambura06\X_test.csv')
        y_train = pd.read_csv(r'C:\Users\user\Omdena\machine-learning-linear-regression-carolynewambura06\y_train.csv')
        y_test = pd.read_csv(r'C:\Users\user\Omdena\machine-learning-linear-regression-carolynewambura06\y_test.csv')
        
        print("Data loaded successfully:")
        print(f"- X_train shape: {X_train.shape}")
        print(f"- X_test shape: {X_test.shape}")
        print(f"- y_train shape: {y_train.shape}")
        print(f"- y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def train_model(X_train, y_train):
    """Train and return a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("\nModel trained successfully.")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"- MSE: {mse:.2f}")
    print(f"- RMSE: {rmse:.2f}")
    print(f"- R² Score: {r2:.2f}")
    
    return y_pred, mse, rmse, r2

def plot_results(y_test, y_pred):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted House Prices')
    plt.savefig('results/actual_vs_predicted.png')
    plt.close()
    print("\nSaved visualization: 'results/actual_vs_predicted.png'")

def main():
    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return
    
    # Step 2: Train model
    model = train_model(X_train, y_train)
    
    # Step 3: Evaluate model
    y_pred, mse, rmse, r2 = evaluate_model(model, X_test, y_test)
    
    # Step 4: Visualize results
    plot_results(y_test, y_pred)
    
    # (Optional) Print coefficients
    print("\nFeature Coefficients:")
    for feature, coef in zip(X_train.columns, model.coef_):
        print(f"- {feature}: {coef:.4f}")

if __name__ == "__main__":
    main()