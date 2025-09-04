#!/usr/bin/env python3
"""
Exact SHAP Implementation for Comparison

This implements exact Kernel SHAP to validate our Low-Rank SHAP implementation.
"""

import numpy as np
from itertools import combinations
from typing import Tuple
import time
from correct_lowrank_shap import CorrectLowRankSHAP


class ExactKernelSHAP:
    """
    Exact Kernel SHAP implementation for comparison.
    
    This computes exact Shapley values using all possible coalitions
    (or a large representative sample for computational feasibility).
    """
    
    def __init__(self, max_coalitions: int = 2000):
        """
        Initialize Exact Kernel SHAP.
        
        Args:
            max_coalitions: Maximum number of coalitions to sample
        """
        self.max_coalitions = max_coalitions
        self.model_fn = None
        self.background_data = None
        self.base_value = None
    
    def _kernel_weight(self, M: int, s: int) -> float:
        """Kernel SHAP weight."""
        if s == 0 or s == M:
            return 1000.0
        return (M - 1) / (s * (M - s))
    
    def _predict_coalition(self, x: np.ndarray, coalition: np.ndarray) -> float:
        """Get model prediction for a coalition."""
        x_modified = x.copy()
        
        if np.sum(coalition) < len(coalition):
            missing_mask = coalition == 0
            x_modified[missing_mask] = np.mean(self.background_data[:, missing_mask], axis=0)
        
        pred = self.model_fn([x_modified])
        if len(pred.shape) > 1 and pred.shape[1] > 1:
            return pred[0, 1] if pred.shape[1] == 2 else pred[0, -1]
        else:
            return pred[0]
    
    def fit(self, model_fn, background_data: np.ndarray, verbose: bool = True):
        """Fit the explainer."""
        self.model_fn = model_fn
        self.background_data = background_data
        
        # Compute base value
        if verbose:
            print("Computing base value...")
        
        base_predictions = []
        for i in range(min(100, len(background_data))):
            pred = self.model_fn([background_data[i]])
            if len(pred.shape) > 1 and pred.shape[1] > 1:
                base_predictions.append(pred[0, 1] if pred.shape[1] == 2 else pred[0, -1])
            else:
                base_predictions.append(pred[0])
        
        self.base_value = np.mean(base_predictions)
        
        if verbose:
            print(f"Exact SHAP base value: {self.base_value:.6f}")
        
        return self
    
    def explain_instance(self, x: np.ndarray) -> np.ndarray:
        """
        Explain a single instance using exact Kernel SHAP.
        
        Args:
            x: Instance to explain
            
        Returns:
            Shapley values
        """
        n_features = len(x)
        
        # Generate all possible coalitions (or sample if too many)
        all_coalitions = []
        all_weights = []
        
        # For small feature sets, use all coalitions
        if n_features <= 10:
            for size in range(n_features + 1):
                for coalition_indices in combinations(range(n_features), size):
                    coalition = np.zeros(n_features)
                    coalition[list(coalition_indices)] = 1
                    all_coalitions.append(coalition)
                    all_weights.append(self._kernel_weight(n_features, size))
        else:
            # For larger feature sets, sample coalitions
            # Always include null and full coalitions
            all_coalitions.append(np.zeros(n_features))
            all_weights.append(self._kernel_weight(n_features, 0))
            
            all_coalitions.append(np.ones(n_features))
            all_weights.append(self._kernel_weight(n_features, n_features))
            
            # Random coalitions
            for _ in range(self.max_coalitions - 2):
                coalition = np.random.binomial(1, 0.5, n_features)
                s = np.sum(coalition)
                all_coalitions.append(coalition)
                all_weights.append(self._kernel_weight(n_features, s))
        
        coalitions = np.array(all_coalitions)
        weights = np.array(all_weights)
        
        # Get predictions
        predictions = []
        for coalition in coalitions:
            pred = self._predict_coalition(x, coalition)
            predictions.append(pred)
        predictions = np.array(predictions)
        
        # Solve weighted linear regression
        X = coalitions.astype(float)
        X_with_intercept = np.column_stack([np.ones(len(coalitions)), X])
        W = np.diag(weights)
        
        try:
            XTW = X_with_intercept.T @ W
            XTWX = XTW @ X_with_intercept
            XTWy = XTW @ predictions
            coefficients = np.linalg.solve(XTWX, XTWy)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(X_with_intercept.T @ W @ X_with_intercept) @ (X_with_intercept.T @ W @ predictions)
        
        return coefficients[1:]  # Skip intercept
    
    def explain(self, X: np.ndarray) -> np.ndarray:
        """Explain multiple instances."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        shapley_values = []
        for x in X:
            shap_vals = self.explain_instance(x)
            shapley_values.append(shap_vals)
        
        return np.array(shapley_values)


def compare_implementations():
    """Compare Low-Rank SHAP with Exact SHAP."""
    print("=== COMPARING LOW-RANK SHAP VS EXACT SHAP ===")
    
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Create synthetic data
    X, y = make_classification(n_samples=200, n_features=4, n_classes=2, 
                              n_informative=3, n_redundant=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=4)
    model.fit(X_train, y_train)
    
    print(f"Model accuracy: {model.score(X_test, y_test):.3f}")
    
    # Setup
    background = X_train[:30]
    test_instances = X_test[:3]
    
    # Test Exact SHAP
    print("\n--- Exact SHAP ---")
    exact_explainer = ExactKernelSHAP(max_coalitions=500)
    exact_explainer.fit(model.predict_proba, background, verbose=True)
    
    start_time = time.time()
    exact_shap = exact_explainer.explain(test_instances)
    exact_time = time.time() - start_time
    
    print(f"Exact SHAP time: {exact_time:.3f}s")
    print(f"Exact SHAP values (instance 0): {exact_shap[0]}")
    
    # Test Low-Rank SHAP
    print("\n--- Low-Rank SHAP ---")
    lr_explainer = CorrectLowRankSHAP(rank=6, n_samples=200)
    lr_explainer.fit(model.predict_proba, background, verbose=True)
    
    start_time = time.time()
    lr_shap = lr_explainer.explain(test_instances)
    lr_time = time.time() - start_time
    
    print(f"Low-Rank SHAP time: {lr_time:.3f}s")
    print(f"Low-Rank SHAP values (instance 0): {lr_shap[0]}")
    
    # Compare accuracy
    print("\n--- Accuracy Comparison ---")
    for i in range(len(test_instances)):
        error = np.linalg.norm(exact_shap[i] - lr_shap[i])
        exact_norm = np.linalg.norm(exact_shap[i])
        relative_error = error / (exact_norm + 1e-10)
        
        print(f"Instance {i}:")
        print(f"  Exact SHAP: {exact_shap[i]}")
        print(f"  Low-Rank SHAP: {lr_shap[i]}")
        print(f"  Absolute error: {error:.6f}")
        print(f"  Relative error: {relative_error:.6f} ({relative_error*100:.2f}%)")
        print(f"  SHAP accuracy: {(1-relative_error)*100:.2f}%")
    
    # Overall statistics
    errors = np.linalg.norm(exact_shap - lr_shap, axis=1)
    exact_norms = np.linalg.norm(exact_shap, axis=1)
    relative_errors = errors / (exact_norms + 1e-10)
    
    print(f"\n--- Overall Performance ---")
    print(f"Mean relative error: {np.mean(relative_errors):.6f} ({np.mean(relative_errors)*100:.2f}%)")
    print(f"Mean SHAP accuracy: {(1-np.mean(relative_errors))*100:.2f}%")
    print(f"Speedup: {exact_time/lr_time:.2f}x")
    
    return {
        'mean_relative_error': np.mean(relative_errors),
        'mean_accuracy': (1-np.mean(relative_errors))*100,
        'speedup': exact_time/lr_time
    }


if __name__ == "__main__":
    results = compare_implementations()
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"SHAP Accuracy: {results['mean_accuracy']:.2f}%")
    print(f"Relative Error: {results['mean_relative_error']*100:.2f}%")
    print(f"Speedup: {results['speedup']:.2f}x")
