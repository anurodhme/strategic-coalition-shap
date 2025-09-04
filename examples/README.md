# Low-Rank SHAP Examples

This directory contains practical examples demonstrating Low-Rank SHAP applications.

## Available Examples

### `real_world_case_study.py`
Complete real-world application demonstrating Low-Rank SHAP on credit risk assessment:
- Realistic 5,000-sample credit dataset with 12 interpretable features
- Multiple model types (Logistic Regression, Random Forest, Gradient Boosting)
- Comprehensive SHAP interpretability analysis
- Business insights generation
- Scalability demonstration up to 20K samples

**Usage:**
```bash
python real_world_case_study.py
```

**Outputs:**
- `results/credit_risk_dataset.csv` - Generated realistic credit dataset
- `results/credit_risk_shap_analysis.csv` - SHAP analysis results
- `results/credit_risk_scalability.csv` - Scalability benchmark results

## Running Examples

All examples are self-contained and include:
- Data generation/loading
- Model training
- Low-Rank SHAP analysis
- Results visualization and interpretation
- Performance benchmarking

Results are automatically saved to the `results/` directory for reproducibility.
