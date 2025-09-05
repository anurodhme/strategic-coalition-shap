#!/usr/bin/env python3
"""
Comprehensive test suite for Low-Rank SHAP package.
Tests core functionality, edge cases, and integration scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from strategic_coalition_shap import StrategicCoalitionSHAP, KernelSHAPBaseline, benchmark_comparison
from strategic_coalition_shap.data_utils import load_wine_data


class TestStrategicCoalitionSHAP:
    """Test suite for StrategicCoalitionSHAP class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample classification data for testing."""
        X, y = make_classification(
            n_samples=200, 
            n_features=10, 
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Train a sample model for testing."""
        X, y = sample_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    def test_initialization(self):
        """Test StrategicCoalitionSHAP initialization with various parameters."""
        # Default initialization
        explainer = StrategicCoalitionSHAP()
        assert explainer.rank == 10
        assert explainer.verbose == True
        
        # Custom parameters
        explainer = StrategicCoalitionSHAP(rank=5, verbose=False)
        assert explainer.rank == 5
        assert explainer.verbose == False
        
        # Invalid rank should raise error
        with pytest.raises(ValueError):
            StrategicCoalitionSHAP(rank=0)
        
        with pytest.raises(ValueError):
            StrategicCoalitionSHAP(rank=-1)
    
    def test_fit(self, sample_data, trained_model):
        """Test fitting the explainer with background data."""
        X, y = sample_data
        model = trained_model
        
        explainer = StrategicCoalitionSHAP(rank=5)
        
        # Test successful fit
        explainer.fit(model, X[:50])
        
        # Check that internal components are set
        assert hasattr(explainer, 'model')
        assert hasattr(explainer, 'X_background')
        assert hasattr(explainer, 'kernel_matrix')
        assert hasattr(explainer, 'U_k')
        assert hasattr(explainer, 'S_k')
        assert hasattr(explainer, 'V_k')
        
        # Check dimensions
        assert explainer.X_background.shape[0] == 50
        assert explainer.U_k.shape[1] == 5  # rank
        assert explainer.S_k.shape[0] == 5  # rank
        assert explainer.V_k.shape[0] == 5  # rank
    
    def test_explain_instance(self, sample_data, trained_model):
        """Test explaining a single instance."""
        X, y = sample_data
        model = trained_model
        
        explainer = StrategicCoalitionSHAP(rank=5)
        explainer.fit(model, X[:50])
        
        # Explain a single instance
        shap_values, metadata = explainer.explain_instance(X[0])
        
        # Check output format
        assert isinstance(shap_values, np.ndarray)
        assert len(shap_values) == X.shape[1]  # One value per feature
        assert isinstance(metadata, dict)
        
        # Check metadata contents
        assert 'runtime' in metadata
        assert 'memory_mb' in metadata
        assert metadata['runtime'] > 0
        assert metadata['memory_mb'] > 0
    
    def test_explain_dataset(self, sample_data, trained_model):
        """Test explaining multiple instances."""
        X, y = sample_data
        model = trained_model
        
        explainer = StrategicCoalitionSHAP(rank=5)
        explainer.fit(model, X[:50])
        
        # Explain multiple instances
        test_instances = X[100:105]  # 5 instances
        shap_values, metadata = explainer.explain_dataset(test_instances)
        
        # Check output format
        assert isinstance(shap_values, np.ndarray)
        assert shap_values.shape == (5, X.shape[1])  # 5 instances, n_features
        assert isinstance(metadata, dict)
        
        # Check metadata
        assert 'total_runtime' in metadata
        assert 'avg_runtime' in metadata
        assert 'max_memory' in metadata
        assert metadata['total_instances'] == 5
    
    def test_different_ranks(self, sample_data, trained_model):
        """Test explainer with different rank values."""
        X, y = sample_data
        model = trained_model
        
        ranks = [3, 5, 10, 15]
        
        for rank in ranks:
            explainer = StrategicCoalitionSHAP(rank=rank, verbose=True)
            explainer.fit(model, X[:50])
            
            shap_values, metadata = explainer.explain_instance(X[0])
            
            # Should work for all ranks
            assert len(shap_values) == X.shape[1]
            assert metadata['runtime'] > 0
    
    def test_edge_cases(self, sample_data, trained_model):
        """Test edge cases and error conditions."""
        X, y = sample_data
        model = trained_model
        
        explainer = StrategicCoalitionSHAP(rank=5)
        
        # Test explaining before fitting
        with pytest.raises(AttributeError):
            explainer.explain_instance(X[0])
        
        # Test with very small background set
        explainer.fit(model, X[:5])  # Only 5 background samples
        shap_values, metadata = explainer.explain_instance(X[0])
        assert len(shap_values) == X.shape[1]
        
        # Test with rank larger than background size
        explainer = StrategicCoalitionSHAP(rank=10)
        explainer.fit(model, X[:5])  # rank > n_background
        shap_values, metadata = explainer.explain_instance(X[0])
        assert len(shap_values) == X.shape[1]


class TestKernelSHAPBaseline:
    """Test suite for KernelSHAPBaseline class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        X, y = make_classification(
            n_samples=100, 
            n_features=5,  # Smaller for exact SHAP
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Train a sample model for testing."""
        X, y = sample_data
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        return model
    
    def test_baseline_functionality(self, sample_data, trained_model):
        """Test basic baseline SHAP functionality."""
        X, y = sample_data
        model = trained_model
        
        baseline = KernelSHAPBaseline()
        baseline.fit(model, X[:20])
        
        shap_values, metadata = baseline.explain_instance(X[0])
        
        assert isinstance(shap_values, np.ndarray)
        assert len(shap_values) == X.shape[1]
        assert isinstance(metadata, dict)
        assert 'runtime' in metadata


class TestBenchmarkComparison:
    """Test suite for benchmark comparison functionality."""
    
    @pytest.fixture
    def sample_setup(self):
        """Create sample setup for benchmarking."""
        X, y = make_classification(
            n_samples=150,
            n_features=8,
            random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return model, X, y
    
    def test_benchmark_comparison(self, sample_setup):
        """Test benchmark comparison between exact and low-rank SHAP."""
        model, X, y = sample_setup
        
        # Run benchmark comparison
        results = benchmark_comparison(
            model=model,
            X_background=X[:30],
            X_test=X[100:105],  # 5 test instances
            ranks=[5, 10],
            verbose=False
        )
        
        # Check results structure
        assert 'exact' in results
        assert 'low_rank' in results
        
        # Check exact results
        exact_results = results['exact']
        assert 'shap_values' in exact_results
        assert 'metadata' in exact_results
        
        # Check low-rank results
        lr_results = results['low_rank']
        for rank in [5, 10]:
            assert rank in lr_results
            assert 'shap_values' in lr_results[rank]
            assert 'metadata' in lr_results[rank]
            assert 'error' in lr_results[rank]
            assert 'relative_error' in lr_results[rank]


class TestDataUtils:
    """Test suite for data utility functions."""
    
    def test_load_wine_data(self):
        """Test wine data loading."""
        try:
            X, y, feature_names = load_wine_data()
            
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert isinstance(feature_names, list)
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] == len(feature_names)
            
        except FileNotFoundError:
            # Data file might not be available in test environment
            pytest.skip("Wine data file not available")


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self):
        """Test complete workflow from data loading to explanation."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            random_state=42
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X_train, y_train)
        
        # Create explainer
        explainer = StrategicCoalitionSHAP(rank=8)
        explainer.fit(model, X_train[:50])
        
        # Explain test instances
        shap_values, metadata = explainer.explain_dataset(X_test[:10])
        
        # Verify results
        assert shap_values.shape == (10, X.shape[1])
        assert metadata['total_instances'] == 10
        assert metadata['total_runtime'] > 0
        
        # Test individual explanation
        single_shap, single_meta = explainer.explain_instance(X_test[0])
        assert len(single_shap) == X.shape[1]
        assert single_meta['runtime'] > 0


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
