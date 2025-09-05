#!/usr/bin/env python3
"""
Exact Kernel SHAP Implementation

Implements exact Kernel SHAP for small problems to provide ground truth
comparison with Low-Rank SHAP. Only feasible for problems with ≤15 features.
"""

import numpy as np
from itertools import combinations
import time


class ExactKernelSHAP:
    """
    Exact Kernel SHAP implementation for ground truth comparison.
    
    Only use for small problems (≤15 features) due to exponential complexity.
    """
    
    def __init__(self, max_coalitions=4096):
        self.max_coalitions = max_coalitions
        self.model_fn = None
        self.background_data = None
        self.base_value = None
    
    def fit(self, model_fn, background_data, verbose=True):
        """Fit the explainer with model and background data."""
        self.model_fn = model_fn
        self.background_data = background_data
        
        # Compute base value
        base_predictions = self.model_fn(background_data)
        if len(base_predictions.shape) > 1 and base_predictions.shape[1] > 1:
            self.base_value = np.mean(base_predictions[:, 1])  # Positive class
        else:
            self.base_value = np.mean(base_predictions)
        
        if verbose:
            print(f"ExactKernelSHAP fitted with base value: {self.base_value:.4f}")
    
    def explain(self, X):
        """Compute exact SHAP values using all possible coalitions."""
        if X.shape[0] != 1:
            raise ValueError("Exact SHAP only supports single instance explanation")
        
        x = X[0]
        n_features = len(x)
        
        if 2**n_features > self.max_coalitions:
            raise ValueError(f"Too many coalitions: 2^{n_features} > {self.max_coalitions}")
        
        # Generate all possible coalitions
        coalitions = []
        weights = []
        predictions = []
        
        for size in range(n_features + 1):
            for coalition in combinations(range(n_features), size):
                coalition_mask = np.zeros(n_features, dtype=bool)
                coalition_mask[list(coalition)] = True
                
                # Kernel SHAP weight
                if size == 0 or size == n_features:
                    weight = 1000  # High weight for boundary coalitions
                else:
                    weight = (n_features - 1) / (size * (n_features - size))
                
                coalitions.append(coalition_mask)
                weights.append(weight)
                
                # Predict with coalition
                pred = self._predict_with_coalition(x, coalition_mask)
                predictions.append(pred)
        
        # Solve weighted least squares
        coalitions = np.array(coalitions, dtype=float)
        weights = np.array(weights)
        predictions = np.array(predictions)
        
        # Add intercept column
        X_matrix = np.column_stack([np.ones(len(coalitions)), coalitions])
        
        # Weighted least squares: (X^T W X)^(-1) X^T W y
        W = np.diag(weights)
        XtWX = X_matrix.T @ W @ X_matrix
        XtWy = X_matrix.T @ W @ predictions
        
        coefficients = np.linalg.solve(XtWX, XtWy)
        
        # SHAP values are the feature coefficients
        shap_values = coefficients[1:]  # Skip intercept
        
        return shap_values.reshape(1, -1)
    
    def _predict_with_coalition(self, x, coalition_mask):
        """Predict with a specific coalition of features."""
        # Create instance with coalition features from x, others from background mean
        background_mean = np.mean(self.background_data, axis=0)
        coalition_instance = background_mean.copy()
        coalition_instance[coalition_mask] = x[coalition_mask]
        
        prediction = self.model_fn(coalition_instance.reshape(1, -1))
        
        if len(prediction.shape) > 1 and prediction.shape[1] > 1:
            return prediction[0, 1]  # Positive class probability
        else:
            return prediction[0]


def validate_exact_vs_lowrank():
    """
    Validate Low-Rank SHAP against Exact SHAP on small problems.
    """
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from strategic_coalition_shap import StrategicCoalitionSHAP
    
    print("=== EXACT SHAP VALIDATION ===")
    
    results = []
    
    for n_features in [8, 10, 12]:
        print(f"\nTesting {n_features} features...")
        
        # Create dataset
        X, y = make_classification(n_samples=500, n_features=n_features,
                                 n_informative=min(n_features-2, 8),
                                 n_redundant=1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        background = X_train[:30]
        test_instance = X_test[:1]
        
        try:
            # Exact SHAP
            exact_explainer = ExactKernelSHAP(max_coalitions=2**n_features)
            exact_explainer.fit(model.predict_proba, background, verbose=False)
            
            start_time = time.time()
            exact_shap = exact_explainer.explain(test_instance)
            exact_time = time.time() - start_time
            
            print(f"  Exact SHAP: {exact_time:.3f}s")
            
            # Low-Rank SHAP at different ranks
            for rank in [5, 8, 10]:
                if rank >= n_features:
                    continue
                
                lr_explainer = StrategicCoalitionSHAP(rank=rank, random_state=42)
                lr_explainer.fit(model.predict_proba, background, verbose=False)
                
                start_time = time.time()
                lr_shap = lr_explainer.explain(test_instance)
                lr_time = time.time() - start_time
                
                # Calculate accuracy
                mse = np.mean((exact_shap - lr_shap)**2)
                mae = np.mean(np.abs(exact_shap - lr_shap))
                relative_error = mae / (np.mean(np.abs(exact_shap)) + 1e-8)
                accuracy = max(0, 1 - relative_error)
                speedup = exact_time / lr_time if lr_time > 0 else 0
                
                results.append({
                    'n_features': n_features,
                    'rank': rank,
                    'exact_time': exact_time,
                    'lr_time': lr_time,
                    'speedup': speedup,
                    'mse': mse,
                    'mae': mae,
                    'accuracy_percent': accuracy * 100
                })
                
                print(f"  Rank {rank}: {accuracy*100:.1f}% accuracy, {speedup:.1f}x speedup")
        
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    return results


if __name__ == "__main__":
    results = validate_exact_vs_lowrank()
    
    if results:
        import pandas as pd
        import os
        os.makedirs('results', exist_ok=True)
        pd.DataFrame(results).to_csv('results/exact_validation_results.csv', index=False)
        print(f"\n✅ Saved {len(results)} validation results to results/exact_validation_results.csv")
