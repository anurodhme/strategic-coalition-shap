#!/usr/bin/env python3
"""
Simple Benchmark: Validate Low-Rank SHAP Performance Claims

This creates verified, reproducible performance metrics for our claims.
Focus: accuracy, runtime, and basic memory usage across core models.
"""

import numpy as np
import pandas as pd
import time
import warnings
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from lowrank_shap import LowRankSHAP
from exact_shap_comparison import ExactKernelSHAP

# Suppress warnings for clean output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.neural_network')


def create_test_data(random_state=42):
    """Create consistent test data."""
    X, y = make_classification(
        n_samples=400, n_features=5, n_classes=2,
        n_informative=4, n_redundant=1, 
        random_state=random_state
    )
    return train_test_split(X, y, test_size=0.3, random_state=random_state)


def benchmark_single_model(model_name, model, X_train, X_test, y_train, y_test, rank=10):
    """Benchmark a single model configuration."""
    print(f"\n--- Benchmarking {model_name} (rank={rank}) ---")
    
    # Train model
    model.fit(X_train, y_train)
    model_accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {model_accuracy:.3f}")
    
    # Setup
    background = X_train[:50]
    test_instances = X_test[:5]
    
    # Test Exact SHAP
    try:
        exact_explainer = ExactKernelSHAP(max_coalitions=200)
        exact_explainer.fit(model.predict_proba, background, verbose=False)
        
        start_time = time.time()
        exact_shap = exact_explainer.explain(test_instances)
        exact_time = time.time() - start_time
        
        print(f"Exact SHAP time: {exact_time:.3f}s")
        
    except Exception as e:
        print(f"❌ Exact SHAP failed: {e}")
        return None
    
    # Test Low-Rank SHAP
    try:
        lr_explainer = LowRankSHAP(rank=rank, random_state=42)
        lr_explainer.fit(model.predict_proba, background, verbose=False)
        
        start_time = time.time()
        lr_shap = lr_explainer.explain(test_instances)
        lr_time = time.time() - start_time
        
        print(f"Low-Rank SHAP time: {lr_time:.3f}s")
        
    except Exception as e:
        print(f"❌ Low-Rank SHAP failed: {e}")
        return None
    
    # Calculate accuracy
    errors = np.linalg.norm(exact_shap - lr_shap, axis=1)
    exact_norms = np.linalg.norm(exact_shap, axis=1)
    relative_errors = errors / (exact_norms + 1e-10)
    
    mean_relative_error = np.mean(relative_errors)
    shap_accuracy = (1 - mean_relative_error) * 100
    speedup = exact_time / lr_time if lr_time > 0 else 0
    
    print(f"SHAP Accuracy: {shap_accuracy:.1f}%")
    print(f"Speedup: {speedup:.2f}x")
    
    return {
        'model': model_name,
        'rank': rank,
        'model_accuracy': model_accuracy,
        'shap_accuracy': shap_accuracy,
        'relative_error': mean_relative_error,
        'speedup': speedup,
        'exact_time': exact_time,
        'lr_time': lr_time,
        'n_features': X_train.shape[1],
        'n_test': len(test_instances)
    }


def run_simple_benchmark():
    """Run simple benchmark across core models."""
    print("=== SIMPLE LOW-RANK SHAP BENCHMARK ===")
    print("Validating performance claims with verified results...\n")
    
    # Create test data
    X_train, X_test, y_train, y_test = create_test_data()
    print(f"Dataset: {X_train.shape[1]} features, {len(X_train)} train samples")
    
    # Test models (fixed configurations to avoid warnings)
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=20, random_state=42, max_depth=5),
        'MLP': MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(30,), early_stopping=True)
    }
    
    results = []
    
    # Test different ranks
    for rank in [5, 10, 15]:
        print(f"\n{'='*50}")
        print(f"TESTING RANK {rank}")
        print(f"{'='*50}")
        
        for model_name, model in models.items():
            result = benchmark_single_model(
                model_name, model, X_train, X_test, y_train, y_test, rank
            )
            if result:
                results.append(result)
    
    return pd.DataFrame(results)


def analyze_results(df):
    """Analyze results and generate verified claims."""
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    print(f"Total successful experiments: {len(df)}")
    
    # Overall statistics
    mean_shap_accuracy = df['shap_accuracy'].mean()
    min_shap_accuracy = df['shap_accuracy'].min()
    mean_speedup = df['speedup'].mean()
    experiments_above_90 = len(df[df['shap_accuracy'] >= 90])
    experiments_with_speedup = len(df[df['speedup'] > 1.0])
    
    print(f"\nOverall Performance:")
    print(f"  Mean SHAP Accuracy: {mean_shap_accuracy:.1f}%")
    print(f"  Min SHAP Accuracy: {min_shap_accuracy:.1f}%")
    print(f"  Mean Speedup: {mean_speedup:.2f}x")
    print(f"  Experiments ≥90% accuracy: {experiments_above_90}/{len(df)}")
    print(f"  Experiments with speedup: {experiments_with_speedup}/{len(df)}")
    
    # Model-specific results
    print(f"\nModel-Specific Results:")
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        print(f"  {model}:")
        print(f"    Mean SHAP Accuracy: {model_df['shap_accuracy'].mean():.1f}%")
        print(f"    Mean Speedup: {model_df['speedup'].mean():.2f}x")
    
    # Generate verified claims
    print(f"\n{'='*60}")
    print("VERIFIED PERFORMANCE CLAIMS")
    print(f"{'='*60}")
    
    claims = []
    
    if mean_shap_accuracy >= 85:
        claims.append(f"✅ Achieves {mean_shap_accuracy:.1f}% average SHAP accuracy")
    
    if min_shap_accuracy >= 80:
        claims.append(f"✅ Maintains ≥{min_shap_accuracy:.1f}% accuracy across all tests")
    
    if experiments_with_speedup > len(df) * 0.5:
        claims.append(f"✅ Provides speedup in {experiments_with_speedup}/{len(df)} experiments")
    
    if mean_speedup > 1.0:
        claims.append(f"✅ Average {mean_speedup:.2f}x speedup over exact SHAP")
    
    for claim in claims:
        print(claim)
    
    if not claims:
        print("❌ No verified performance claims can be made based on current results")
    
    return {
        'mean_shap_accuracy': mean_shap_accuracy,
        'min_shap_accuracy': min_shap_accuracy,
        'mean_speedup': mean_speedup,
        'verified_claims': claims
    }


def main():
    """Run the simple benchmark."""
    # Run benchmark
    results_df = run_simple_benchmark()
    
    # Analyze results
    stats = analyze_results(results_df)
    
    # Save results
    output_file = '/Users/anurodhbudhathoki/New Analysis/results/simple_benchmark_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to {output_file}")
    
    return results_df, stats


if __name__ == "__main__":
    results, stats = main()
