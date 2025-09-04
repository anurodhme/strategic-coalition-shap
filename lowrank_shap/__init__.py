"""Low-Rank SHAP: Efficient Model-Agnostic Feature Attribution via Randomized SVD.

This package provides fast approximation of Shapley values using low-rank SVD decomposition
of the kernel matrix, reducing computational complexity from O(n²) to O(nk).

Main Classes:
    LowRankSHAP: Core low-rank SHAP implementation
    KernelSHAPBaseline: Baseline exact Kernel SHAP for comparison

Utility Functions:
    load_wine_data: Load wine quality dataset
    load_adult_data: Load adult income dataset  
    load_compas_data: Load COMPAS dataset
    benchmark_comparison: Compare exact vs low-rank SHAP performance

Example:
    >>> from lowrank_shap import LowRankSHAP
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    
    >>> # Generate sample data
    >>> X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    >>> model = RandomForestClassifier(random_state=42)
    >>> model.fit(X, y)
    
    >>> # Create and fit Low-Rank SHAP explainer
    >>> explainer = LowRankSHAP(rank=10)
    >>> explainer.fit(model, X[:100])  # Use 100 background samples
    
    >>> # Explain a single instance
    >>> shap_values, metadata = explainer.explain_instance(X[0])
    >>> print(f"SHAP values: {shap_values}")
    >>> print(f"Runtime: {metadata['runtime']:.3f}s")
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# Core implementations
from .lowrank_shap import LowRankSHAP
from .baseline import KernelSHAPBaseline

# Data utilities
from .data_utils import (
    load_wine_quality,
    load_adult,
    load_compas,
    load_bike_sharing
)

# Benchmarking utilities
from .benchmark import benchmark_comparison

# Export main API
__all__ = [
    "LowRankSHAP",
    "KernelSHAPBaseline", 
    "load_wine_quality",
    "load_adult",
    "load_compas",
    "load_bike_sharing",
    "benchmark_comparison",
]