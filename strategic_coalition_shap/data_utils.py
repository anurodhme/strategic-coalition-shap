"""
Data loading utilities for the lowrank-shap project.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_wine_quality():
    """Load and prepare wine quality dataset."""
    # Load with semicolon separator
    df = pd.read_csv('data/raw/wine.csv', sep=';')
    
    # Features and target
    X = df.drop('quality', axis=1).values
    y = df['quality'].values
    
    return X, y, df.columns.tolist()


def load_bike_sharing():
    """Load and prepare bike sharing dataset."""
    # Try multiple strategies to load the CSV
    strategies = [
        # Strategy 1: Default UTF-8
        {'encoding': 'utf-8'},
        # Strategy 2: Latin-1 encoding
        {'encoding': 'latin-1'},
        # Strategy 3: Different separator
        {'encoding': 'utf-8', 'sep': ';'},
        {'encoding': 'latin-1', 'sep': ';'},
        # Strategy 4: Tab separated
        {'encoding': 'utf-8', 'sep': '\t'},
        # Strategy 5: Python engine with auto-detection
        {'engine': 'python', 'sep': None},
        # Strategy 6: Skip bad lines
        {'encoding': 'utf-8', 'on_bad_lines': 'skip'},
        {'encoding': 'latin-1', 'on_bad_lines': 'skip'},
    ]
    
    df = None
    for i, strategy in enumerate(strategies):
        try:
            df = pd.read_csv('data/raw/bike.csv', **strategy)
            print(f"  Bike dataset loaded with strategy {i+1}: {df.shape}")
            break
        except Exception as e:
            if i == len(strategies) - 1:  # Last strategy
                raise Exception(f"All loading strategies failed. Last error: {e}")
            continue
    
    # Basic preprocessing for bike sharing
    # Target: cnt (total rental count)
    if 'cnt' in df.columns:
        target_col = 'cnt'
    elif 'count' in df.columns:
        target_col = 'count'
    else:
        # Assume last column is target
        target_col = df.columns[-1]
    
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values
    
    return X, y, df.columns.tolist()


def load_adult():
    """Load and prepare adult income dataset."""
    df = pd.read_csv('data/raw/adult.csv')
    
    # Basic preprocessing for categorical variables
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Target: income (assuming last column)
    target_col = df.columns[-1]
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values
    
    return X, y, df.columns.tolist()


def load_compas():
    """Load and prepare COMPAS dataset."""
    df = pd.read_csv('data/raw/compas.csv')
    
    # Handle missing values first
    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
    
    # Basic preprocessing for categorical variables
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Target: assume last column
    target_col = df.columns[-1]
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values
    
    # Final check for any remaining NaN values
    if np.isnan(X).any():
        print(f"  Warning: Remaining NaN values in features, filling with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    if np.isnan(y).any():
        print(f"  Warning: NaN values in target, removing affected samples")
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
    
    return X, y, df.columns.tolist()


def prepare_dataset(X, y, test_size=0.2, random_state=42, max_samples=None):
    """
    Prepare dataset for experiments with scaling and subsampling.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set
        random_state: Random seed
        max_samples: Maximum samples to use (for testing)
    
    Returns:
        Dictionary with train/test splits and scaler
    """
    # Subsample if requested
    if max_samples and len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'n_features': X.shape[1],
        'n_samples': len(X)
    }


def get_small_datasets(max_samples=1000):
    """
    Get small datasets for baseline testing.
    
    Args:
        max_samples: Maximum samples per dataset
    
    Returns:
        Dictionary with prepared datasets
    """
    datasets = {}
    
    # Wine Quality (smallest)
    try:
        X, y, cols = load_wine_quality()
        datasets['wine'] = prepare_dataset(X, y, max_samples=max_samples)
        datasets['wine']['columns'] = cols
    except Exception as e:
        print(f"Error loading wine dataset: {e}")
    
    # Bike Sharing
    try:
        X, y, cols = load_bike_sharing()
        datasets['bike'] = prepare_dataset(X, y, max_samples=max_samples)
        datasets['bike']['columns'] = cols
    except Exception as e:
        print(f"Error loading bike dataset: {e}")
    
    return datasets
