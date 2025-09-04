"""
Kernel SHAP Baseline Implementation
Week 1 Task: Reproduce standard Kernel SHAP for baseline measurements
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_kernels
from typing import Optional, Tuple
import time
import psutil
import os


class KernelSHAPBaseline:
    """
    Standard Kernel SHAP implementation for baseline measurements.
    
    This implements the exact Kernel SHAP algorithm as described in:
    Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
    """
    
    def __init__(self, kernel_width: float = 0.25, n_samples: int = 2048):
        """
        Initialize Kernel SHAP baseline.
        
        Args:
            kernel_width: Width parameter for the kernel
            n_samples: Number of samples for the Kernel SHAP approximation
        """
        self.kernel_width = kernel_width
        self.n_samples = n_samples
        self.model = None
        self.background_data = None
        self.kernel_matrix = None
        
    def fit(self, model: BaseEstimator, background_data: np.ndarray):
        """
        Fit the Kernel SHAP explainer.
        
        Args:
            model: Trained sklearn model to explain
            background_data: Background dataset for reference
        """
        self.model = model
        self.background_data = background_data
        
    def _kernel_shap_kernel(self, M: int, s: int) -> float:
        """
        Compute the Kernel SHAP kernel weight.
        
        Args:
            M: Total number of features
            s: Number of features in the subset
            
        Returns:
            Kernel weight for the subset
        """
        if s == 0 or s == M:
            return 100000  # Large weight for empty/full sets
        
        return (M - 1) / (s * (M - s))
    
    def _generate_coalitions(self, n_features: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random feature coalitions for Kernel SHAP.
        
        Args:
            n_features: Number of features
            
        Returns:
            Tuple of (coalition_matrix, kernel_weights)
        """
        # Generate random subsets
        coalition_matrix = np.random.choice([0, 1], 
                                          size=(self.n_samples, n_features), 
                                          p=[0.5, 0.5])
        
        # Ensure we have empty and full sets
        coalition_matrix[0, :] = 0  # Empty set
        coalition_matrix[1, :] = 1  # Full set
        
        # Compute kernel weights
        subset_sizes = np.sum(coalition_matrix, axis=1)
        kernel_weights = np.array([self._kernel_shap_kernel(n_features, s) 
                                 for s in subset_sizes])
        
        return coalition_matrix, kernel_weights
    
    def _predict_with_coalition(self, 
                              x: np.ndarray, 
                              coalition: np.ndarray) -> float:
        """
        Make prediction with a given feature coalition.
        
        Args:
            x: Input instance to explain
            coalition: Binary vector indicating which features to use
            
        Returns:
            Model prediction for the coalition
        """
        # Create modified instance
        x_modified = x.copy()
        
        # For features not in coalition, use background data
        if coalition.sum() < len(coalition):
            # Sample random background instances
            bg_indices = np.random.choice(len(self.background_data), 
                                        size=100, replace=True)
            bg_sample = self.background_data[bg_indices]
            
            # Replace features not in coalition with background values
            mask = coalition == 0
            x_modified[mask] = np.mean(bg_sample[:, mask], axis=0)
        
        return self.model.predict([x_modified])[0]
    
    def explain_instance(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Explain a single instance using Kernel SHAP.
        
        Args:
            x: Instance to explain (1D array)
            
        Returns:
            Tuple of (shapley_values, metadata)
        """
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        n_features = len(x)
        
        # Generate coalitions
        coalition_matrix, kernel_weights = self._generate_coalitions(n_features)
        
        # Get predictions for all coalitions
        predictions = []
        for coalition in coalition_matrix:
            pred = self._predict_with_coalition(x, coalition)
            predictions.append(pred)
        predictions = np.array(predictions)
        
        # Solve weighted least squares problem
        # y = X @ phi + epsilon, where X is coalition_matrix
        # Weighted by kernel_weights
        
        # Create design matrix
        X = coalition_matrix.astype(float)
        
        # Weighted least squares
        W = np.diag(kernel_weights)
        WX = W @ X
        Wy = W @ predictions
        
        # Solve (X^T W X) phi = X^T W y
        try:
            shapley_values = np.linalg.solve(X.T @ WX, X.T @ Wy)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if singular
            shapley_values = np.linalg.pinv(X.T @ WX) @ X.T @ Wy
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        metadata = {
            'runtime': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'n_features': n_features,
            'n_samples': self.n_samples
        }
        
        return shapley_values, metadata
    
    def explain_dataset(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Explain multiple instances.
        
        Args:
            X: Dataset to explain (2D array)
            
        Returns:
            Tuple of (shapley_values_matrix, metadata)
        """
        shapley_values = []
        runtimes = []
        memory_usages = []
        
        for i, x in enumerate(X):
            sv, meta = self.explain_instance(x)
            shapley_values.append(sv)
            runtimes.append(meta['runtime'])
            memory_usages.append(meta['memory_usage'])
            
            if i % 10 == 0:
                print(f"Explained {i+1}/{len(X)} instances")
        
        metadata = {
            'total_runtime': sum(runtimes),
            'avg_runtime': np.mean(runtimes),
            'max_memory': max(memory_usages),
            'total_instances': len(X)
        }
        
        return np.array(shapley_values), metadata


def benchmark_kernel_shap(model, X_background, X_test, n_samples=2048):
    """
    Benchmark Kernel SHAP on test data.
    
    Args:
        model: Trained sklearn model
        X_background: Background dataset
        X_test: Test instances to explain
        n_samples: Number of samples for Kernel SHAP
        
    Returns:
        Benchmark results dictionary
    """
    explainer = KernelSHAPBaseline(n_samples=n_samples)
    explainer.fit(model, X_background)
    
    shap_values, metadata = explainer.explain_dataset(X_test)
    
    return {
        'shap_values': shap_values,
        'metadata': metadata
    }
