"""
Low-Rank SHAP Implementation
Week 2 Task: Implement low-rank approximation for Kernel SHAP
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator
from typing import Optional, Tuple, Dict, Any
import time
import psutil
import os


class LowRankSHAP:
    """
    Low-rank approximation for Kernel SHAP using SVD.
    
    This implements the efficient computation of Shapley values using
    rank-k SVD approximation of the kernel matrix.
    """
    
    def __init__(self, 
                 rank: int = 50, 
                 kernel_width: float = 0.25,
                 regularization: float = 1e-6):
        """
        Initialize Low-Rank SHAP.
        
        Args:
            rank: Target rank for low-rank approximation
            kernel_width: Width parameter for the kernel
            regularization: Small regularization term for numerical stability
        """
        self.rank = rank
        self.kernel_width = kernel_width
        self.regularization = regularization
        
        # Fitted parameters
        self.model = None
        self.background_data = None
        self.U_k = None
        self.S_k = None
        self.V_k = None
        self.kernel_matrix = None
        
    def _compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix between two sets of points.
        
        Args:
            X1: First set of points (n1 x d)
            X2: Second set of points (n2 x d)
            
        Returns:
            Kernel matrix (n1 x n2)
        """
        # Use RBF kernel
        pairwise_dist = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)
        kernel = np.exp(-pairwise_dist / (2 * self.kernel_width**2))
        return kernel
    
    def fit(self, 
            model: BaseEstimator, 
            background_data: np.ndarray,
            verbose: bool = False) -> 'LowRankSHAP':
        """
        Fit the low-rank SHAP explainer.
        
        Args:
            model: Trained sklearn model
            background_data: Background dataset for reference
            verbose: Whether to print progress
            
        Returns:
            self: Fitted explainer
        """
        self.model = model
        self.background_data = background_data
        
        if verbose:
            print("Computing kernel matrix...")
        
        # Compute kernel matrix
        self.kernel_matrix = self._compute_kernel_matrix(
            background_data, background_data
        )
        
        if verbose:
            print(f"Kernel matrix shape: {self.kernel_matrix.shape}")
            print("Computing low-rank SVD...")
        
        # Compute low-rank SVD
        try:
            # Use randomized SVD for efficiency
            U_k, S_k, V_k = svds(self.kernel_matrix, k=self.rank)
            
            # svds returns singular values in ascending order
            # Reverse to get descending order
            self.U_k = U_k[:, ::-1]
            self.S_k = S_k[::-1]
            self.V_k = V_k[::-1, :]
            
            # Add regularization
            self.S_k += self.regularization
            
        except Exception as e:
            raise ValueError(f"SVD computation failed: {e}")
        
        if verbose:
            print(f"SVD completed. Rank: {self.rank}")
            print(f"Singular values: {self.S_k[:5]}...")
        
        return self
    
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
            return 100000.0
        
        return (M - 1) / (s * (M - s))
    
    def _generate_coalitions(self, n_features: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random feature coalitions.
        
        Args:
            n_features: Number of features
            n_samples: Number of samples
            
        Returns:
            Tuple of (coalition_matrix, kernel_weights)
        """
        # Generate random subsets
        coalition_matrix = np.random.choice([0, 1], 
                                          size=(n_samples, n_features), 
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
    
    def explain_instance(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Explain a single instance using low-rank SHAP.
        
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
        coalition_matrix, kernel_weights = self._generate_coalitions(
            n_features, n_samples=2048
        )
        
        # Get predictions for all coalitions
        predictions = []
        for coalition in coalition_matrix:
            pred = self._predict_with_coalition(x, coalition)
            predictions.append(pred)
        predictions = np.array(predictions)
        
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
            shapley_values = np.linalg.pinv(X.T @ WX) @ X.T @ Wy
        
        # Apply low-rank correction using kernel matrix
        # This is where the low-rank approximation comes into play
        
        # Compute kernel vector between x and background
        kernel_vec = self._compute_kernel_matrix(
            x.reshape(1, -1), self.background_data
        ).flatten()
        
        # Apply low-rank correction
        # phi_corrected = V_k Σ_k^{-1} U_k^T kernel_vec * shapley_values
        if self.U_k is not None:
            # Low-rank correction factor
            correction = self.V_k.T @ np.diag(1.0 / self.S_k) @ self.U_k.T @ kernel_vec
            shapley_values = shapley_values * correction[:n_features]
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        metadata = {
            'runtime': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'n_features': n_features,
            'rank': self.rank,
            'kernel_matrix_shape': self.kernel_matrix.shape if self.kernel_matrix is not None else None
        }
        
        return shapley_values, metadata
    
    def explain_dataset(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Explain multiple instances using low-rank SHAP.
        
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
            
            if i % 5 == 0:
                print(f"Explained {i+1}/{len(X)} instances")
        
        metadata = {
            'total_runtime': sum(runtimes),
            'avg_runtime': np.mean(runtimes),
            'max_memory': max(memory_usages),
            'total_instances': len(X),
            'rank': self.rank
        }
        
        return np.array(shapley_values), metadata


def benchmark_comparison(model, 
                        background_data: np.ndarray,
                        test_data: np.ndarray,
                        ranks: list = [10, 30, 50],
                        exact_samples: int = 2048) -> Dict[str, Any]:
    """
    Benchmark comparison between exact and low-rank SHAP.
    
    Args:
        model: Trained sklearn model
        background_data: Background dataset
        test_data: Test instances to explain
        ranks: List of ranks to test
        exact_samples: Samples for exact Kernel SHAP
        
    Returns:
        Benchmark results dictionary
    """
    from lowrank_shap.baseline import KernelSHAPBaseline
    
    results = {
        'exact': None,
        'low_rank': {}
    }
    
    # Test on small subset for comparison
    n_test = min(10, len(test_data))
    X_test_small = test_data[:n_test]
    
    # Exact Kernel SHAP
    print("Running exact Kernel SHAP...")
    exact_explainer = KernelSHAPBaseline(n_samples=exact_samples)
    exact_explainer.fit(model, background_data)
    exact_shap, exact_meta = exact_explainer.explain_dataset(X_test_small)
    
    results['exact'] = {
        'shap_values': exact_shap,
        'metadata': exact_meta
    }
    
    # Low-rank SHAP for different ranks
    for rank in ranks:
        print(f"Running low-rank SHAP (rank={rank})...")
        lowrank_explainer = LowRankSHAP(rank=rank)
        lowrank_explainer.fit(model, background_data, verbose=False)
        lowrank_shap, lowrank_meta = lowrank_explainer.explain_dataset(X_test_small)
        
        # Compute error metrics
        error = np.linalg.norm(exact_shap - lowrank_shap, axis=1)
        relative_error = error / (np.linalg.norm(exact_shap, axis=1) + 1e-8)
        
        results['low_rank'][rank] = {
            'shap_values': lowrank_shap,
            'metadata': lowrank_meta,
            'error': error,
            'relative_error': relative_error,
            'mean_relative_error': np.mean(relative_error)
        }
    
    return results
