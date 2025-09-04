#!/usr/bin/env python3
"""
Debug script for Week 3 experiments.
Tests individual components to isolate issues.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowrank_shap.data_utils import load_wine_quality, load_bike_sharing, load_adult, load_compas
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lowrank_shap.lowrank_shap import LowRankSHAP, benchmark_comparison
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """Test data loading functions."""
    print("=== Testing Data Loading ===")
    
    datasets = {
        'wine': load_wine_quality,
        'bike': load_bike_sharing, 
        'adult': load_adult,
        'compas': load_compas
    }
    
    for name, loader in datasets.items():
        try:
            print(f"\nTesting {name}...")
            X, y, columns = loader()
            print(f"  ✓ Success: {X.shape[0]} samples, {X.shape[1]} features")
        except Exception as e:
            print(f"  ✗ Error: {e}")

def test_svd_robustness():
    """Test SVD with different ranks and datasets."""
    print("\n=== Testing SVD Robustness ===")
    
    # Load wine dataset (known to work)
    X, y, _ = load_wine_quality()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Test different ranks
    background = X_train_scaled[:100]  # Small background
    test_instances = X_test_scaled[:3]  # Very small test set
    
    for rank in [5, 10, 20, 30]:
        try:
            print(f"\nTesting rank {rank}...")
            explainer = LowRankSHAP(rank=rank)
            explainer.fit(model, background, verbose=False)
            shap_values, metadata = explainer.explain_dataset(test_instances)
            print(f"  ✓ Success: SHAP values shape {shap_values.shape}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

def test_benchmark_function():
    """Test the benchmark comparison function."""
    print("\n=== Testing Benchmark Function ===")
    
    # Load wine dataset
    X, y, _ = load_wine_quality()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Test benchmark with conservative settings
    background = X_train_scaled[:50]  # Small background
    test_instances = X_test_scaled[:2]  # Very small test set
    
    try:
        print("Running benchmark with ranks [5, 10]...")
        results = benchmark_comparison(
            model, background, test_instances, ranks=[5, 10]
        )
        print("  ✓ Benchmark completed successfully")
        
        # Print summary
        exact_time = results['exact']['metadata']['total_runtime']
        exact_memory = results['exact']['metadata']['max_memory']
        
        for rank in [5, 10]:
            if str(rank) in results['low_rank']:
                lr_results = results['low_rank'][str(rank)]
                speedup = exact_time / lr_results['metadata']['total_runtime']
                memory_ratio = exact_memory / lr_results['metadata']['max_memory']
                error = lr_results['mean_relative_error']
                print(f"  Rank {rank}: {speedup:.1f}x speed, {memory_ratio:.1f}x memory, error={error:.4f}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    test_data_loading()
    test_svd_robustness()
    test_benchmark_function()
    print("\n=== Debug Complete ===")
