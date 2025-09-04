#!/usr/bin/env python3
"""
Diagnostic script to identify the root cause of Low-Rank SHAP accuracy issues.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
import os

# Add the package to path
sys.path.insert(0, '.')

print("=== DIAGNOSTIC TEST: LOW-RANK SHAP IMPLEMENTATION ISSUES ===")

# Create simple, controlled dataset
np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=4, n_classes=2, 
                          n_informative=3, n_redundant=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train simple model
model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
model.fit(X_train, y_train)

print(f"Model accuracy: {model.score(X_test, y_test):.3f}")
print(f"Dataset shape: {X.shape}")

# Use small background set for testing
X_background = X_train[:20]  # Small background set
X_explain = X_test[:2]       # Few instances to explain

print(f"Background samples: {len(X_background)}")
print(f"Test instances: {len(X_explain)}")

# Test 1: Check if we can import and instantiate
print("\n--- TEST 1: Import and Instantiation ---")
try:
    from lowrank_shap import LowRankSHAP
    print("✅ LowRankSHAP import successful")
    
    # Try to import KernelSHAP from different locations
    try:
        from lowrank_shap import KernelSHAP
        print("✅ KernelSHAP import from main package successful")
    except ImportError:
        try:
            from lowrank_shap.kernel_shap import KernelSHAPBaseline as KernelSHAP
            print("✅ KernelSHAP import from kernel_shap module successful")
        except ImportError:
            # Create a simple exact SHAP implementation for comparison
            print("⚠️ KernelSHAP not found, will create simple baseline")
            KernelSHAP = None
    
    # Create explainers
    lr_explainer = LowRankSHAP(rank=3)
    print("✅ LowRankSHAP instantiation successful")
    
    if KernelSHAP is not None:
        exact_explainer = KernelSHAP()
        print("✅ Exact explainer instantiation successful")
    else:
        exact_explainer = None
        print("⚠️ Will skip exact SHAP comparison")
    
except Exception as e:
    print(f"❌ Import/instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Check fitting process
print("\n--- TEST 2: Fitting Process ---")
try:
    # Fit low-rank SHAP
    print("Fitting Low-Rank SHAP...")
    lr_explainer.fit(model.predict_proba, X_background, verbose=True)
    print("✅ Low-rank SHAP fit successful")
    
    # Check if SVD was computed
    print(f"SVD components shape: U_k={lr_explainer.U_k.shape if lr_explainer.U_k is not None else None}")
    print(f"Singular values: {lr_explainer.S_k}")
    print(f"Kernel matrix shape: {lr_explainer.kernel_matrix.shape if lr_explainer.kernel_matrix is not None else None}")
    
    # Fit exact SHAP if available
    if exact_explainer is not None:
        exact_explainer.fit(model.predict_proba, X_background)
        print("✅ Exact SHAP fit successful")
    
except Exception as e:
    print(f"❌ Fitting failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Check explanation process
print("\n--- TEST 3: Explanation Process ---")
try:
    # Get low-rank SHAP values  
    print("Computing Low-Rank SHAP explanations...")
    lr_shap, lr_metadata = lr_explainer.explain_dataset(X_explain)
    print(f"✅ Low-rank SHAP shape: {lr_shap.shape}")
    print(f"Low-rank SHAP values (instance 0): {lr_shap[0]}")
    print(f"Low-rank SHAP metadata: {lr_metadata}")
    
    # Check for suspicious values
    if np.any(np.isnan(lr_shap)):
        print("❌ NaN values detected in Low-rank SHAP!")
    if np.any(np.isinf(lr_shap)):
        print("❌ Infinite values detected in Low-rank SHAP!")
    if np.all(np.abs(lr_shap) < 1e-10):
        print("❌ All SHAP values are essentially zero!")
    
    # Get exact SHAP values if available
    exact_shap = None
    if exact_explainer is not None:
        try:
            exact_shap = exact_explainer.explain(X_explain)
            print(f"✅ Exact SHAP shape: {exact_shap.shape}")
            print(f"Exact SHAP values (instance 0): {exact_shap[0]}")
        except Exception as e:
            print(f"⚠️ Exact SHAP failed: {e}")
    else:
        print("⚠️ Skipping exact SHAP (not available)")
    
except Exception as e:
    print(f"❌ Explanation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Calculate actual accuracy (if exact SHAP available)
print("\n--- TEST 4: Accuracy Analysis ---")
try:
    if exact_shap is not None:
        # Calculate relative error
        error = np.linalg.norm(exact_shap - lr_shap, axis=1)
        exact_norm = np.linalg.norm(exact_shap, axis=1)
        relative_error = error / (exact_norm + 1e-10)
        
        print(f"Absolute error per instance: {error}")
        print(f"Exact SHAP norm per instance: {exact_norm}")
        print(f"Relative error per instance: {relative_error}")
        print(f"Mean relative error: {np.mean(relative_error):.6f}")
        print(f"SHAP accuracy: {(1 - np.mean(relative_error)) * 100:.6f}%")
        
        # Check if values are reasonable
        if np.mean(relative_error) > 0.1:
            print("❌ HIGH RELATIVE ERROR DETECTED! (>10%)")
        else:
            print("✅ Relative error is acceptable (<10%)")
    else:
        print("⚠️ Cannot calculate accuracy without exact SHAP")
        
    # Basic sanity checks on Low-Rank SHAP values
    print(f"\nLow-Rank SHAP sanity checks:")
    print(f"  Value range: [{np.min(lr_shap):.6f}, {np.max(lr_shap):.6f}]")
    print(f"  Mean absolute value: {np.mean(np.abs(lr_shap)):.6f}")
    print(f"  Sum per instance: {np.sum(lr_shap, axis=1)}")
    
    # Check for problematic values
    if np.any(np.isnan(lr_shap)) or np.any(np.isinf(lr_shap)):
        print("❌ NaN or infinite values detected in Low-rank SHAP!")
    else:
        print("✅ No NaN or infinite values")
        
except Exception as e:
    print(f"❌ Accuracy analysis failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Investigate the correction mechanism
print("\n--- TEST 5: Correction Mechanism Analysis ---")
try:
    # Let's manually trace through the correction logic
    instance = X_explain[0]
    
    # Check kernel computation
    kernel_vec = lr_explainer._compute_kernel_matrix(
        instance.reshape(1, -1), lr_explainer.background_data
    ).flatten()
    print(f"Kernel vector shape: {kernel_vec.shape}")
    print(f"Kernel vector values: {kernel_vec[:5]}...")
    
    # Check SVD correction computation
    if lr_explainer.U_k is not None:
        correction = lr_explainer.V_k.T @ np.diag(1.0 / lr_explainer.S_k) @ lr_explainer.U_k.T @ kernel_vec
        print(f"Correction vector shape: {correction.shape}")
        print(f"Correction values: {correction}")
        
        # This is likely where the bug is!
        print(f"Number of features: {len(instance)}")
        print(f"Correction vector length: {len(correction)}")
        
        if len(correction) != len(instance):
            print("❌ DIMENSION MISMATCH: Correction vector doesn't match feature count!")
        else:
            print("✅ Dimensions match")
            
except Exception as e:
    print(f"❌ Correction analysis failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DIAGNOSTIC COMPLETE ===")
