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
