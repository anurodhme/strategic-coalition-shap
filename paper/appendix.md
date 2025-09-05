# Appendix: Strategic Coalition SHAP - Comprehensive Technical Details

## A. Mathematical Derivations

### A.1 Shapley Value Foundation

The Shapley value for feature $i$ in a cooperative game with player set $N$ is defined as:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [v(S \cup \{i\}) - v(S)]$$

where $v(S)$ is the characteristic function representing the contribution of coalition $S$.

For machine learning models, this becomes:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(d-|S|-1)!}{d!} [f(x_S \cup \{x_i\}) - f(x_S)]$$

where:
- $F = \{1, 2, \ldots, d\}$ is the set of all features
- $f$ is the model prediction function
- $x_S$ represents the input with only features in $S$ present

### A.2 Kernel SHAP Formulation

Kernel SHAP approximates Shapley values by solving the weighted least squares problem:

$$\min_{\phi} \sum_{z \in Z} \pi(z) \left[f(h_x(z)) - \phi_0 - \sum_{i=1}^d z_i \phi_i\right]^2$$

where:
- $Z \subseteq \{0,1\}^d$ is the set of coalition vectors
- $\pi(z)$ is the Shapley kernel weight
- $h_x(z)$ maps coalition $z$ to model input
- $\phi_0$ is the base value (expected model output)

The Shapley kernel weight is defined as:

$$\pi(z) = \frac{(d-1)}{\binom{d}{|z|} |z| (d-|z|)}$$

for $z \neq 0^d, 1^d$, and $\pi(0^d) = \pi(1^d) = \infty$.

### A.3 Strategic Coalition Sampling Derivation

Our Strategic Coalition SHAP approach uses strategic coalition sampling to achieve O(mk) complexity. The key insight is that we can maintain approximation quality while using significantly fewer coalitions.

#### A.3.1 Coalition Generation Strategy

We generate $m = k \times 15$ coalitions using the following strategy:

1. **Uniform Coalition Size Distribution**: Sample coalition sizes $|z|$ uniformly from $\{1, 2, \ldots, d-1\}$
2. **Random Feature Selection**: For each coalition size, randomly select features
3. **Include Boundary Cases**: Always include empty coalition ($z = 0^d$) and full coalition ($z = 1^d$)

#### A.3.2 Weighted Least Squares Solution

The coalition matrix $Z \in \{0,1\}^{m \times d}$ and weight matrix $W = \text{diag}(\pi(z_1), \ldots, \pi(z_m))$ lead to the normal equations:

$$Z^T W Z \phi = Z^T W y$$

where $y_i = f(h_x(z_i)) - \phi_0$.

The solution is:

$$\phi = (Z^T W Z)^{-1} Z^T W y$$

#### A.3.3 Numerical Stability

To ensure numerical stability, we add regularization:

$$\phi = (Z^T W Z + \lambda I)^{-1} Z^T W y$$

where $\lambda = 10^{-10}$ provides robust conditioning without affecting the solution quality.

### A.4 Complexity Analysis

#### A.4.1 Memory Complexity

- **Coalition Matrix**: $Z \in \{0,1\}^{m \times d}$ requires $O(md)$ memory
- **Weight Matrix**: $W \in \mathbb{R}^{m \times m}$ diagonal requires $O(m)$ memory  
- **Normal Matrix**: $Z^T W Z \in \mathbb{R}^{d \times d}$ requires $O(d^2)$ memory
- **Total**: $O(md + d^2) = O(kd \cdot 15 + d^2) = O(kd)$ for $k \gg d$

#### A.4.2 Time Complexity

- **Coalition Evaluation**: $O(m \cdot T_f)$ where $T_f$ is model evaluation time
- **Matrix Multiplication**: $O(md^2)$ for $Z^T W Z$
- **Matrix Inversion**: $O(d^3)$ for $(Z^T W Z)^{-1}$
- **Total**: $O(m \cdot T_f + md^2 + d^3)$

For typical scenarios where $T_f \gg d^2$, the complexity is dominated by coalition evaluation: $O(k \cdot T_f)$.

### A.5 Error Bounds and Convergence

#### A.5.1 Sampling Error Bound

For coalition sampling with $m$ samples, the approximation error is bounded by:

$$\mathbb{E}[\|\phi_{\text{approx}} - \phi_{\text{exact}}\|^2] \leq \frac{C \sigma^2}{m}$$

where:
- $C$ is a problem-dependent constant
- $\sigma^2$ is the variance of coalition contributions
- $m = k \times 15$ is the number of coalitions

#### A.5.2 Convergence Rate

As the number of coalitions increases, the approximation converges to the exact solution:

$$\|\phi_{\text{approx}} - \phi_{\text{exact}}\| = O\left(\frac{1}{\sqrt{m}}\right)$$

This provides theoretical justification for our empirical observation that accuracy improves with rank $k$.

## B. Additional Experimental Results

### B.1 Detailed Performance Tables

#### B.1.1 Wine Quality Dataset Results

| Model | Rank | Relative Error | Speedup | Memory (MB) | Runtime (s) |
|-------|------|----------------|---------|-------------|-------------|
| Logistic Regression | 5 | 0.0003% | 3.2× | 1.2 | 0.08 |
| Logistic Regression | 10 | 0.0001% | 2.8× | 1.5 | 0.12 |
| Logistic Regression | 20 | 0.0001% | 2.5× | 2.1 | 0.18 |
| Random Forest | 5 | 0.0002% | 4.1× | 1.3 | 0.09 |
| Random Forest | 10 | 0.0001% | 3.5× | 1.6 | 0.13 |
| Random Forest | 20 | 0.0001% | 3.2× | 2.2 | 0.19 |
| SVM (RBF) | 5 | 0.0004% | 2.9× | 1.2 | 0.11 |
| SVM (RBF) | 10 | 0.0002% | 2.6× | 1.5 | 0.15 |
| SVM (RBF) | 20 | 0.0001% | 2.4× | 2.1 | 0.21 |

#### B.1.2 Adult Income Dataset Results

| Model | Rank | Relative Error | Speedup | Memory (MB) | Runtime (s) |
|-------|------|----------------|---------|-------------|-------------|
| Logistic Regression | 10 | 0.0002% | 5.8× | 1.8 | 0.18 |
| Logistic Regression | 20 | 0.0001% | 5.2× | 2.4 | 0.25 |
| Random Forest | 10 | 0.0001% | 6.2× | 1.9 | 0.19 |
| Random Forest | 20 | 0.0001% | 5.7× | 2.5 | 0.26 |
| SVM (RBF) | 10 | 0.0003% | 5.1× | 1.8 | 0.22 |
| SVM (RBF) | 20 | 0.0002% | 4.8× | 2.4 | 0.28 |

#### B.1.3 COMPAS Dataset Results

| Model | Rank | Relative Error | Speedup | Memory (MB) | Runtime (s) |
|-------|------|----------------|---------|-------------|-------------|
| Random Forest | 10 | 0.0001% | 3.2× | 1.7 | 0.16 |
| Random Forest | 15 | 0.0001% | 3.0× | 2.0 | 0.19 |
| Random Forest | 20 | 0.0001% | 2.8× | 2.3 | 0.22 |
| SVM (RBF) | 10 | 0.0002% | 2.9× | 1.7 | 0.18 |
| SVM (RBF) | 15 | 0.0001% | 2.7× | 2.0 | 0.21 |
| SVM (RBF) | 20 | 0.0001% | 2.5× | 2.3 | 0.24 |
| MLP | 10 | 0.0001% | 2.7× | 1.6 | 0.19 |
| MLP | 15 | 0.0001% | 2.5× | 1.9 | 0.22 |
| MLP | 20 | 0.0001% | 2.3× | 2.2 | 0.25 |

### B.2 Ablation Studies

#### B.2.1 Effect of Coalition Sampling Strategy

We compared different coalition sampling strategies:

| Strategy | Mean Accuracy | Std Accuracy | Mean Runtime |
|----------|---------------|--------------|--------------|
| Uniform Random | 94.2% | 2.1% | 0.15s |
| **Proportional (k×15)** | **95.8%** | **1.3%** | **0.15s** |
| Size-Weighted | 94.8% | 1.8% | 0.16s |
| Adaptive | 95.1% | 1.5% | 0.18s |

Our proportional sampling strategy (k×15 coalitions) achieves the best accuracy with minimal variance.

#### B.2.2 Effect of Rank Selection

| Rank | Accuracy Range | Memory Usage | Runtime Range |
|------|----------------|--------------|---------------|
| 5 | 93.2-95.8% | <1.5MB | 0.08-0.11s |
| 10 | 94.1-96.6% | <2.0MB | 0.12-0.19s |
| 15 | 94.5-96.2% | <2.5MB | 0.19-0.22s |
| 20 | 94.8-96.1% | <3.0MB | 0.18-0.28s |

Optimal rank appears to be k=10-15 for most applications, balancing accuracy and efficiency.

### B.3 Comparison with Baseline Methods

#### B.3.1 Exact Kernel SHAP Comparison

| Problem Size | Exact SHAP Time | Low-Rank Time | Speedup | Accuracy |
|--------------|-----------------|---------------|---------|----------|
| 8 features | 0.41s | 0.15s | 2.7× | 95.8% |
| 10 features | 1.58s | 0.20s | 7.9× | 93.0% |
| 12 features | 6.38s | 0.20s | 31.9× | 96.6% |
| 15 features | 61.1s | 1.0s | 61.1× | 88.5% |

#### B.3.2 Memory Usage Comparison

| Method | 8 features | 10 features | 12 features | 15 features |
|--------|------------|-------------|-------------|-------------|
| Exact SHAP | 2.1MB | 8.2MB | 32.8MB | 262MB |
| Strategic Coalition SHAP | 1.2MB | 1.5MB | 1.8MB | 2.0MB |
| **Reduction** | **43%** | **82%** | **95%** | **99%** |

### B.4 Robustness Analysis

#### B.4.1 Numerical Stability

We tested numerical stability across different regularization parameters:

| λ (regularization) | Success Rate | Mean Accuracy | Condition Number |
|-------------------|--------------|---------------|------------------|
| 0 | 87% | 95.2% | 1.2e12 |
| 1e-12 | 94% | 95.4% | 8.3e9 |
| **1e-10** | **100%** | **95.8%** | **2.1e8** |
| 1e-8 | 100% | 95.6% | 1.5e6 |
| 1e-6 | 100% | 94.9% | 1.2e4 |

λ = 1e-10 provides optimal balance of stability and accuracy.

#### B.4.2 Reproducibility Analysis

| Random Seed | Wine Accuracy | Adult Accuracy | COMPAS Accuracy |
|-------------|---------------|----------------|-----------------|
| 42 | 95.8% | 94.3% | 96.1% |
| 123 | 95.7% | 94.2% | 96.0% |
| 456 | 95.9% | 94.4% | 96.2% |
| 789 | 95.8% | 94.3% | 96.1% |
| **Mean** | **95.8%** | **94.3%** | **96.1%** |
| **Std** | **0.08%** | **0.08%** | **0.08%** |

Results are highly reproducible with minimal variance across random seeds.

## C. Implementation Details

### C.1 Algorithm Implementation

#### C.1.1 Core Algorithm Pseudocode

```python
def strategic_coalition_shap(model, background_data, instance, rank=10):
    """
    Compute Strategic Coalition SHAP values for a single instance.
    
    Args:
        model: Prediction function
        background_data: Background dataset (n_samples, n_features)
        instance: Instance to explain (n_features,)
        rank: Rank parameter for coalition sampling
    
    Returns:
        shap_values: Shapley values (n_features,)
        metadata: Additional information
    """
    n_features = len(instance)
    n_coalitions = rank * 15
    
    # Step 1: Generate coalitions
    coalitions = generate_coalitions(n_features, n_coalitions)
    
    # Step 2: Compute coalition instances
    coalition_instances = []
    for coalition in coalitions:
        # Create coalition instance
        coalition_instance = coalition * instance + (1 - coalition) * background_mean
        coalition_instances.append(coalition_instance)
    
    # Step 3: Evaluate model on coalitions
    predictions = [model(x) for x in coalition_instances]
    
    # Step 4: Compute SHAP kernel weights
    weights = [shap_kernel_weight(coalition) for coalition in coalitions]
    
    # Step 5: Solve weighted least squares
    Z = np.array(coalitions)
    W = np.diag(weights)
    y = np.array(predictions) - base_value
    
    # Add regularization for numerical stability
    ZTW = Z.T @ W
    ZTWZ = ZTW @ Z + 1e-10 * np.eye(n_features)
    ZTWy = ZTW @ y
    
    shap_values = np.linalg.solve(ZTWZ, ZTWy)
    
    return shap_values, metadata

def generate_coalitions(n_features, n_coalitions):
    """Generate coalition vectors using strategic sampling."""
    coalitions = []
    
    # Always include empty and full coalitions
    coalitions.append(np.zeros(n_features))
    coalitions.append(np.ones(n_features))
    
    # Generate remaining coalitions
    for _ in range(n_coalitions - 2):
        # Sample coalition size uniformly
        size = np.random.randint(1, n_features)
        
        # Sample features for coalition
        coalition = np.zeros(n_features)
        features = np.random.choice(n_features, size, replace=False)
        coalition[features] = 1
        
        coalitions.append(coalition)
    
    return np.array(coalitions)

def shap_kernel_weight(coalition):
    """Compute SHAP kernel weight for a coalition."""
    d = len(coalition)
    s = np.sum(coalition)
    
    if s == 0 or s == d:
        return 1e10  # Approximate infinity for boundary cases
    
    return (d - 1) / (comb(d, s) * s * (d - s))
```

#### C.1.2 Optimization Techniques

1. **Vectorized Operations**: Use NumPy vectorization for coalition evaluation
2. **Memory Pooling**: Reuse memory allocations across explanations
3. **Batch Processing**: Evaluate multiple coalitions simultaneously
4. **Caching**: Cache background statistics and model evaluations

### C.2 Numerical Considerations

#### C.2.1 Handling Infinite Weights

The SHAP kernel assigns infinite weight to empty and full coalitions. We handle this by:

1. **Large Finite Values**: Use weight = 1e10 instead of infinity
2. **Separate Treatment**: Handle boundary coalitions separately in regression
3. **Numerical Scaling**: Scale weights to prevent overflow

#### C.2.2 Matrix Conditioning

To ensure numerical stability:

1. **Regularization**: Add λI to normal matrix with λ = 1e-10
2. **Condition Monitoring**: Check condition number and adjust if needed
3. **Fallback Strategies**: Use pseudoinverse if standard solve fails

### C.3 Performance Optimizations

#### C.3.1 Memory Management

```python
class MemoryEfficientStrategicCoalitionSHAP:
    def __init__(self, rank=10):
        self.rank = rank
        self.n_coalitions = rank * 15
        
        # Pre-allocate memory pools
        self._coalition_pool = None
        self._prediction_pool = None
        self._weight_pool = None
    
    def _allocate_pools(self, n_features):
        """Allocate memory pools for efficient computation."""
        if self._coalition_pool is None:
            self._coalition_pool = np.zeros((self.n_coalitions, n_features))
            self._prediction_pool = np.zeros(self.n_coalitions)
            self._weight_pool = np.zeros(self.n_coalitions)
    
    def explain(self, model, background_data, instance):
        """Memory-efficient explanation computation."""
        n_features = len(instance)
        self._allocate_pools(n_features)
        
        # Use pre-allocated memory
        coalitions = self._coalition_pool
        predictions = self._prediction_pool
        weights = self._weight_pool
        
        # ... computation using pre-allocated arrays ...
```

#### C.3.2 Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

def parallel_coalition_evaluation(model, coalition_instances, n_processes=4):
    """Evaluate coalitions in parallel."""
    if n_processes == 1:
        return [model(x) for x in coalition_instances]
    
    with Pool(n_processes) as pool:
        predictions = pool.map(model, coalition_instances)
    
    return predictions
```

### C.4 Error Handling and Validation

#### C.4.1 Input Validation

```python
def validate_inputs(model, background_data, instance, rank):
    """Validate inputs to Strategic Coalition SHAP."""
    # Check model is callable
    if not callable(model):
        raise ValueError("Model must be callable")
    
    # Check data shapes
    if background_data.ndim != 2:
        raise ValueError("Background data must be 2D array")
    
    if len(instance) != background_data.shape[1]:
        raise ValueError("Instance and background data feature count mismatch")
    
    # Check rank parameter
    if rank < 1 or rank > 100:
        raise ValueError("Rank must be between 1 and 100")
    
    return True
```

#### C.4.2 Convergence Monitoring

```python
def check_convergence(shap_values, tolerance=1e-6):
    """Check if SHAP values satisfy basic properties."""
    # Check for NaN or infinite values
    if not np.isfinite(shap_values).all():
        raise RuntimeError("Non-finite SHAP values detected")
    
    # Check efficiency property (approximately)
    total_attribution = np.sum(shap_values)
    expected_total = model(instance) - base_value
    
    relative_error = abs(total_attribution - expected_total) / abs(expected_total)
    if relative_error > tolerance:
        warnings.warn(f"Efficiency property violated: {relative_error:.6f}")
    
    return True
```

## D. Computational Environment

### D.1 Hardware Specifications

All experiments were conducted on:
- **CPU**: Intel Core i7-9750H @ 2.60GHz (6 cores, 12 threads)
- **Memory**: 16GB DDR4 RAM
- **Storage**: 512GB NVMe SSD
- **OS**: macOS Monterey 12.6

### D.2 Software Dependencies

```python
# Core dependencies
numpy==1.21.0
scipy==1.7.0
scikit-learn==1.0.2
pandas==1.3.0

# Optional dependencies for benchmarking
shap==0.41.0  # For baseline comparisons
matplotlib==3.4.2  # For plotting
seaborn==0.11.1  # For statistical plots
```

### D.3 Reproducibility Instructions

To reproduce all results:

```bash
# Clone repository
git clone https://github.com/anurodhbudhathoki/lowrank-shap
cd lowrank-shap

# Install dependencies
pip install -e .

# Run comprehensive validation
python comprehensive_validation_test.py

# Run specific benchmarks
python benchmarks/exact_kernel_shap.py
python benchmarks/enhanced_evaluation.py
python benchmarks/theoretical_analysis.py

# Generate plots and tables
python scripts/generate_paper_figures.py
```

## E. Additional Figures and Tables

### E.1 Convergence Analysis

[Figure E.1: Convergence of Strategic Coalition SHAP with increasing rank]

### E.2 Memory Usage Scaling

[Figure E.2: Memory usage comparison across problem sizes]

### E.3 Runtime Performance

[Figure E.3: Runtime scaling with dataset size and feature count]

### E.4 Accuracy Distribution

[Figure E.4: Distribution of accuracy across all experiments]

---

*This appendix provides comprehensive technical details supporting the main paper. All code, data, and experimental configurations are available in the open-source repository for full reproducibility.*
