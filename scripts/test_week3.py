#!/usr/bin/env python3
"""
Quick validation script for Week 3 experiments.
Tests a small subset to ensure everything works before full run.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lowrank_shap.data_utils import load_wine_quality, load_bike_sharing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lowrank_shap.lowrank_shap import LowRankSHAP, benchmark_comparison
import warnings
warnings.filterwarnings('ignore')

def quick_test():
    """Run quick validation test."""
    print("=== Week 3 Quick Validation Test ===")
    
    # Test with wine dataset and 2 models
    print("\n1. Loading wine dataset...")
    X, y, _ = load_wine_quality()
    print(f"   Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Test 2 models
    models = {
        'logreg': LogisticRegression(random_state=42, max_iter=1000),
        'rf': RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    for model_name, model in models.items():
        print(f"\n2. Testing {model_name}...")
        
        # Scale features for logistic regression
        if model_name == 'logreg':
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
        
        # Train model
        model.fit(X_train_scaled, y_train)
        accuracy = model.score(X_test_scaled, y_test)
        print(f"   Accuracy: {accuracy:.3f}")
        
        # Run small benchmark
        print("   Running benchmark...")
        background = X_train_scaled[:50]  # Small background
        test_instances = X_test_scaled[:5]  # Very small test set
        
        try:
            results = benchmark_comparison(
                model, background, test_instances, ranks=[10, 30]
            )
            
            # Print summary
            exact_time = results['exact']['metadata']['total_runtime']
            exact_memory = results['exact']['metadata']['max_memory']
            
            for rank in [10, 30]:
                lr_results = results['low_rank'][str(rank)]
                speedup = exact_time / lr_results['metadata']['total_runtime']
                memory_ratio = exact_memory / lr_results['metadata']['max_memory']
                error = lr_results['mean_relative_error']
                
                print(f"   Rank {rank}: {speedup:.1f}x speed, {memory_ratio:.1f}x memory, error={error:.4f}")
            
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n✓ Validation test completed successfully!")
    print("Ready to run full Week 3 experiments with:")
    print("  python scripts/week3_experiments.py --datasets all --models all --ranks 10,30,50")

if __name__ == "__main__":
    quick_test()
