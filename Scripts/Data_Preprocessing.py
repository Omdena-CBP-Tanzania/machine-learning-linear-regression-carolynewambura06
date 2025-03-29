# scripts/data_preprocessing.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Preprocess the data:
    1. One-Hot Encode 'rad' column
    2. Separate features (X) and target (y)
    3. Standardize features
    """
    # One-Hot Encoding for 'rad'
    df = pd.get_dummies(df, columns=['rad'], prefix='rad', dtype=int)
    
    # Split features and target
    X = df.drop(columns=['medv'])
    y = df['medv']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def main():
    # Load data
    data_path = r"C:\Users\user\Omdena\machine-learning-linear-regression-carolynewambura06\BostonHousing.csv"
    df = load_data(data_path)
    
    # Initial checks
    print("Missing values:\n", df.isnull().sum())
    print("\nData info:")
    df.info()
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Output shapes
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Optionally: Save processed data
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    
    print("\nPreprocessing complete. Data saved to 'data/processed/'.")

if __name__ == "__main__":
    main()