# Strategic Coalition SHAP: Mathematical Derivation

## Problem Setup

Given a kernel matrix $K \in \mathbb{R}^{n \times n}$ from Kernel SHAP, we want to approximate Shapley values efficiently using low-rank SVD.

## Standard Kernel SHAP

The Shapley values $\phi \in \mathbb{R}^n$ are computed as:

$$\phi = K^{-1} y$$

where:
- $K$ is the kernel matrix with $K_{ij} = k(x_i, x_j)$
- $y$ is the vector of model predictions
- $n$ is the number of training instances

## Low-Rank Approximation

We approximate the kernel matrix using rank-$k$ SVD:

$$K \approx U_k \Sigma_k V_k^\top$$

where:
- $U_k \in \mathbb{R}^{n \times k}$ contains the left singular vectors
- $\Sigma_k \in \mathbb{R}^{k \times k}$ is diagonal with singular values
- $V_k \in \mathbb{R}^{n \times k}$ contains the right singular vectors

## Closed-Form Update

Using the low-rank approximation, the Shapley values become:

$$\phi_{\text{low-rank}} = V_k \Sigma_k^{-1} U_k^\top y$$

### Derivation Steps

1. **Start with exact solution**:
   $$\phi = K^{-1} y$$

2. **Substitute low-rank approximation**:
   $$\phi \approx (U_k \Sigma_k V_k^\top)^{-1} y$$

3. **Use pseudoinverse property**:
   $$(U_k \Sigma_k V_k^\top)^{-1} = V_k \Sigma_k^{-1} U_k^\top$$

4. **Final form**:
   $$\phi_{\text{low-rank}} = V_k \Sigma_k^{-1} U_k^\top y$$

## Computational Complexity

| Method | Time Complexity | Memory Complexity |
|--------|----------------|------------------|
| Exact Kernel SHAP | $O(n^3)$ | $O(n^2)$ |
| Strategic Coalition SHAP | $O(mk^2 + mk)$ | $O(mk)$ |

## Error Analysis

The approximation error is bounded by:

$$\|\phi - \phi_{\text{low-rank}}\|_2 \leq \frac{\sigma_{k+1}}{\sigma_k} \|y\|_2$$

where $\sigma_k$ is the $k$-th singular value.

## Implementation Notes

1. **Randomized SVD**: Use `scipy.sparse.linalg.svds` for efficient computation
2. **Memory optimization**: Store only $U_k$, $\Sigma_k$, $V_k$ instead of full $K$
3. **Numerical stability**: Add small regularization to diagonal of $\Sigma_k$

## Algorithm

```python
def lowrank_shap(K, y, k):
    # Compute rank-k SVD
    U_k, S_k, V_k = svds(K, k=k)
    
    # Compute inverse of diagonal
    S_k_inv = np.diag(1.0 / S_k)
    
    # Compute low-rank Shapley values
    phi = V_k @ S_k_inv @ U_k.T @ y
    
    return phi
```
