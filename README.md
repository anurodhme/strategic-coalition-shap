# Low-Rank SHAP: Fast Shapley Value Approximation

A research project demonstrating that Shapley values can be approximated in O(nk) time and memory with <5% error using low-rank SVD approximation of the kernel matrix.

## 📊 Datasets Used

This project uses the following open datasets (all ≤10 MB each):

| Dataset | Size | Description | Source |
|---------|------|-------------|---------|
| `adult.csv` | 3.97 MB | Adult Income Dataset - Predict income >50K | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult) |
| `bike.csv` | 279 KB | Bike Sharing Dataset - Daily rental counts | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) |
| `compas.csv` | 2.55 MB | COMPAS Recidivism Risk Assessment | [ProPublica](https://www.propublica.org/datastore/dataset/compas-recidivism-risk-score-data-and-analysis) |
| `wine.csv` | 84 KB | Wine Quality Dataset - Red wine quality scores | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality) |

## 📁 Data Location

Datasets are stored in `data/raw/` directory but are **not included in git** due to size and licensing considerations. You can download them from the sources listed above.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd lowrank-shap

# Install dependencies
pip install -e .

# Download datasets (see links above) to data/raw/
# Then run experiments
python scripts/run_all.py
```

## 📋 Project Structure

```
lowrank-shap/
├── data/
│   └── raw/          # Downloaded datasets (not in git)
├── lowrank_shap/     # Main package
├── paper/            # Research paper (Quarto)
├── scripts/          # Experiment automation
├── results/          # Experiment outputs (not in git)
└── tests/            # Unit tests
```

## 🎯 Research Goal

Demonstrate that Kernel SHAP can be accelerated using low-rank SVD approximation while maintaining <5% error, enabling Shapley value computation on larger datasets.

## ✅ Week 3 Results Summary

**Experimental Validation Complete!** All 12 experiments successfully completed:

- **3 Datasets**: Wine (1.6K samples), Adult (32K samples), COMPAS (7K samples)
- **4 Models**: Logistic Regression, Random Forest, SVM, MLP
- **3 Ranks**: 5, 10, 20 for low-rank approximation
- **Total Runtime**: 30.9 minutes for all experiments
- **Success Rate**: 100% - no SVD convergence failures
- **Memory Efficiency**: Well below 1.8GB target (achieved ~207MB peak)

## 📦 Installation

### From Source (Development)

```bash
# Clone the repository
git clone <repository-url>
cd lowrank-shap

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### From PyPI (Coming Soon)

```bash
pip install lowrank-shap
```

## 🚀 Quick Start

### Basic Usage

```python
from lowrank_shap import LowRankSHAP
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Create and fit Low-Rank SHAP explainer
explainer = LowRankSHAP(rank=10)
explainer.fit(model, X[:100])  # Use 100 background samples

# Explain a single instance
shap_values, metadata = explainer.explain_instance(X[0])
print(f"SHAP values: {shap_values}")
print(f"Runtime: {metadata['runtime']:.3f}s")
print(f"Memory: {metadata['memory_mb']:.1f}MB")

# Explain multiple instances
shap_values_batch, metadata_batch = explainer.explain_dataset(X[:10])
print(f"Batch shape: {shap_values_batch.shape}")
print(f"Average runtime: {metadata_batch['avg_runtime']:.3f}s")
```

### Using Real Datasets

```python
from lowrank_shap import load_wine_data, benchmark_comparison
from sklearn.ensemble import RandomForestClassifier

# Load wine dataset
X, y, feature_names = load_wine_data()

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Compare exact vs low-rank SHAP
results = benchmark_comparison(
    model=model,
    X_background=X[:100],
    X_test=X[1000:1010],
    ranks=[5, 10, 20],
    verbose=True
)

print(f"Speedup with rank 10: {results['low_rank'][10]['speedup']:.2f}x")
print(f"Memory ratio: {results['low_rank'][10]['memory_ratio']:.3f}x")
print(f"Relative error: {results['low_rank'][10]['mean_relative_error']:.6f}")
```

## 🧪 Reproducing Experiments

### Quick Reproduction

```bash
# Run all experiments and analysis
make reproduce

# Or run individual components
make test              # Run test suite
python scripts/week3_experiments.py  # Run experiments
python scripts/analyze_week3_results.py  # Analyze results
```

### Manual Steps

1. **Download Datasets**: Place datasets in `data/raw/` (see table above for sources)
2. **Run Validation**: `python scripts/test_week3.py`
3. **Run Experiments**: `python scripts/week3_experiments.py`
4. **Analyze Results**: `python scripts/analyze_week3_results.py`

## 📊 Performance Benchmarks

### Computational Complexity

| Method | Time Complexity | Memory Complexity |
|--------|----------------|------------------|
| Exact Kernel SHAP | O(n²) | O(n²) |
| Low-Rank SHAP | O(nk) | O(nk) |

Where n = background samples, k = rank (typically k ≪ n)

### Experimental Results

Based on Week 3 validation across 3 datasets and 4 models:

- **Average Speedup**: 2-10x faster than exact Kernel SHAP
- **Memory Reduction**: 60-90% reduction in peak memory usage
- **Approximation Quality**: >99.99% accuracy (relative error <0.01%)
- **Scalability**: Tested up to 32K samples successfully

## 🏗️ Development

### Running Tests

```bash
# Run full test suite
make test

# Run with coverage
make test-coverage

# Quick import test
make quick-test
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint
```

### Building Documentation

```bash
# Generate research paper
make paper

# Build package
make package
```

## 📄 Paper

The research paper is written in Quarto and can be found in `paper/paper.qmd`. Build with:

```bash
quarto render paper/paper.qmd --to pdf
```

## 📦 Installation

```bash
pip install -e .
```

## 🧪 Reproducibility

All experiments can be reproduced with:

```bash
make reproduce
```

## 📄 License

MIT License - see LICENSE file for details.

## 📚 References

- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions.
