#!/usr/bin/env python3
"""
Benchmarking utilities for comparing exact Kernel SHAP vs Low-Rank SHAP.
Provides comprehensive performance comparison including runtime, memory, and accuracy metrics.
"""

import time
import numpy as np
import psutil
import os
from typing import Dict, List, Any, Tuple, Optional

from .baseline import KernelSHAPBaseline
from .lowrank_shap import LowRankSHAP


def benchmark_comparison(
    model,
    X_background: np.ndarray,
    X_test: np.ndarray,
    ranks: List[int] = [5, 10, 20],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare exact Kernel SHAP vs Low-Rank SHAP across multiple ranks.
    
    Args:
        model: Trained scikit-learn model
        X_background: Background dataset for SHAP computation
        X_test: Test instances to explain
        ranks: List of ranks to test for low-rank approximation
        verbose: Whether to print progress information
        
    Returns:
        Dictionary containing detailed benchmark results:
        - 'exact': Results from exact Kernel SHAP
        - 'low_rank': Results from Low-Rank SHAP for each rank
    """
    
    if verbose:
        print(f"🔍 Benchmarking SHAP methods...")
        print(f"   Background samples: {X_background.shape[0]}")
        print(f"   Test instances: {X_test.shape[0]}")
        print(f"   Features: {X_background.shape[1]}")
        print(f"   Ranks to test: {ranks}")
    
    results = {
        'exact': {},
        'low_rank': {}
    }
    
    # Benchmark exact Kernel SHAP
    if verbose:
        print("\n📊 Running exact Kernel SHAP...")
    
    exact_explainer = KernelSHAPBaseline()
    exact_explainer.fit(model, X_background)
    
    # Measure exact SHAP performance
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    exact_shap_values = []
    exact_runtimes = []
    exact_memories = []
    
    for i, instance in enumerate(X_test):
        instance_start = time.time()
        shap_vals, metadata = exact_explainer.explain_instance(instance)
        instance_time = time.time() - instance_start
        
        exact_shap_values.append(shap_vals)
        exact_runtimes.append(instance_time)
        exact_memories.append(metadata.get('memory_mb', 0))
        
        if verbose and (i + 1) % max(1, len(X_test) // 4) == 0:
            print(f"   Explained {i + 1}/{len(X_test)} instances")
    
    total_exact_time = time.time() - start_time
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    peak_exact_memory = max(exact_memories) if exact_memories else end_memory - start_memory
    
    exact_shap_values = np.array(exact_shap_values)
    
    results['exact'] = {
        'shap_values': exact_shap_values,
        'metadata': {
            'total_runtime': total_exact_time,
            'avg_runtime': np.mean(exact_runtimes),
            'max_memory': peak_exact_memory,
            'total_instances': len(X_test)
        }
    }
    
    if verbose:
        print(f"   ✅ Exact SHAP completed in {total_exact_time:.2f}s")
        print(f"   📊 Average per-instance: {np.mean(exact_runtimes):.3f}s")
        print(f"   💾 Peak memory: {peak_exact_memory:.1f}MB")
    
    # Benchmark Low-Rank SHAP for each rank
    for rank in ranks:
        if verbose:
            print(f"\n🚀 Running Low-Rank SHAP (rank={rank})...")
        
        lr_explainer = LowRankSHAP(rank=rank, verbose=False)
        lr_explainer.fit(model, X_background)
        
        # Measure low-rank SHAP performance
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        lr_shap_values = []
        lr_runtimes = []
        lr_memories = []
        
        for i, instance in enumerate(X_test):
            instance_start = time.time()
            shap_vals, metadata = lr_explainer.explain_instance(instance)
            instance_time = time.time() - instance_start
            
            lr_shap_values.append(shap_vals)
            lr_runtimes.append(instance_time)
            lr_memories.append(metadata.get('memory_mb', 0))
            
            if verbose and (i + 1) % max(1, len(X_test) // 4) == 0:
                print(f"   Explained {i + 1}/{len(X_test)} instances")
        
        total_lr_time = time.time() - start_time
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        peak_lr_memory = max(lr_memories) if lr_memories else end_memory - start_memory
        
        lr_shap_values = np.array(lr_shap_values)
        
        # Calculate performance metrics
        speedup = results['exact']['metadata']['avg_runtime'] / np.mean(lr_runtimes) if np.mean(lr_runtimes) > 0 else 0
        memory_ratio = peak_lr_memory / results['exact']['metadata']['max_memory'] if results['exact']['metadata']['max_memory'] > 0 else 0
        
        # Calculate approximation error
        error_per_instance = np.linalg.norm(exact_shap_values - lr_shap_values, axis=1)
        relative_error_per_instance = error_per_instance / (np.linalg.norm(exact_shap_values, axis=1) + 1e-10)
        mean_relative_error = np.mean(relative_error_per_instance)
        
        results['low_rank'][rank] = {
            'shap_values': lr_shap_values,
            'metadata': {
                'total_runtime': total_lr_time,
                'avg_runtime': np.mean(lr_runtimes),
                'max_memory': peak_lr_memory,
                'total_instances': len(X_test),
                'rank': rank
            },
            'error': error_per_instance,
            'relative_error': relative_error_per_instance,
            'mean_relative_error': mean_relative_error,
            'speedup': speedup,
            'memory_ratio': memory_ratio
        }
        
        if verbose:
            print(f"   ✅ Low-Rank SHAP (rank={rank}) completed in {total_lr_time:.2f}s")
            print(f"   📊 Average per-instance: {np.mean(lr_runtimes):.3f}s")
            print(f"   💾 Peak memory: {peak_lr_memory:.1f}MB")
            print(f"   🚀 Speedup: {speedup:.2f}x")
            print(f"   💾 Memory ratio: {memory_ratio:.3f}x")
            print(f"   🎯 Mean relative error: {mean_relative_error:.6f}")
    
    if verbose:
        print(f"\n🎉 Benchmark completed!")
        print(f"   Total configurations tested: {1 + len(ranks)}")
        
        # Summary of best performance
        best_speedup_rank = max(ranks, key=lambda r: results['low_rank'][r]['speedup'])
        best_memory_rank = min(ranks, key=lambda r: results['low_rank'][r]['memory_ratio'])
        best_accuracy_rank = min(ranks, key=lambda r: results['low_rank'][r]['mean_relative_error'])
        
        print(f"   🏆 Best speedup: rank {best_speedup_rank} ({results['low_rank'][best_speedup_rank]['speedup']:.2f}x)")
        print(f"   🏆 Best memory: rank {best_memory_rank} ({results['low_rank'][best_memory_rank]['memory_ratio']:.3f}x)")
        print(f"   🏆 Best accuracy: rank {best_accuracy_rank} ({results['low_rank'][best_accuracy_rank]['mean_relative_error']:.6f} error)")
    
    return results


def quick_benchmark(
    model,
    X_background: np.ndarray,
    X_test: np.ndarray,
    rank: int = 10,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Quick benchmark comparing exact vs low-rank SHAP for a single rank.
    
    Args:
        model: Trained scikit-learn model
        X_background: Background dataset
        X_test: Test instances to explain
        rank: Rank for low-rank approximation
        verbose: Whether to print results
        
    Returns:
        Dictionary with speedup, memory_ratio, and mean_relative_error
    """
    
    results = benchmark_comparison(
        model=model,
        X_background=X_background,
        X_test=X_test,
        ranks=[rank],
        verbose=False
    )
    
    metrics = {
        'speedup': results['low_rank'][rank]['speedup'],
        'memory_ratio': results['low_rank'][rank]['memory_ratio'],
        'mean_relative_error': results['low_rank'][rank]['mean_relative_error']
    }
    
    if verbose:
        print(f"🚀 Quick Benchmark Results (rank={rank}):")
        print(f"   Speedup: {metrics['speedup']:.2f}x")
        print(f"   Memory ratio: {metrics['memory_ratio']:.3f}x")
        print(f"   Relative error: {metrics['mean_relative_error']:.6f}")
    
    return metrics


def memory_profile_comparison(
    model,
    X_background: np.ndarray,
    sample_sizes: List[int] = [50, 100, 200, 500],
    rank: int = 10,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Profile memory usage across different background sample sizes.
    
    Args:
        model: Trained scikit-learn model
        X_background: Full background dataset
        sample_sizes: List of background sample sizes to test
        rank: Rank for low-rank approximation
        verbose: Whether to print progress
        
    Returns:
        Dictionary with memory usage for exact and low-rank methods
    """
    
    if verbose:
        print(f"📊 Memory profiling across sample sizes: {sample_sizes}")
    
    exact_memories = []
    lr_memories = []
    
    # Single test instance
    X_test = X_background[:1]
    
    for size in sample_sizes:
        if size > len(X_background):
            if verbose:
                print(f"   ⚠️ Skipping size {size} (exceeds background size {len(X_background)})")
            continue
            
        X_bg_subset = X_background[:size]
        
        if verbose:
            print(f"   Testing with {size} background samples...")
        
        # Test exact SHAP memory
        exact_explainer = KernelSHAPBaseline()
        exact_explainer.fit(model, X_bg_subset)
        _, exact_meta = exact_explainer.explain_instance(X_test[0])
        exact_memories.append(exact_meta.get('memory_mb', 0))
        
        # Test low-rank SHAP memory
        lr_explainer = LowRankSHAP(rank=rank, verbose=False)
        lr_explainer.fit(model, X_bg_subset)
        _, lr_meta = lr_explainer.explain_instance(X_test[0])
        lr_memories.append(lr_meta.get('memory_mb', 0))
        
        if verbose:
            reduction = (1 - lr_memories[-1] / exact_memories[-1]) * 100 if exact_memories[-1] > 0 else 0
            print(f"     Exact: {exact_memories[-1]:.1f}MB, Low-rank: {lr_memories[-1]:.1f}MB ({reduction:.1f}% reduction)")
    
    return {
        'sample_sizes': sample_sizes[:len(exact_memories)],
        'exact_memory': exact_memories,
        'lowrank_memory': lr_memories
    }
