# Literature Gap Summary

1. **SHAP (2017)** – O(n²) RAM; no low-rank variant.
2. **FastSHAP (NeurIPS 2022)** – surrogate NN, not model-agnostic.
3. **Shapley Net (ICML 2023)** – only CNN images; closed source.
4. **SAGE (2023)** – global importance; still O(n²) kernel.
5. **RS-SHAP (KDD 2024)** – sampling for speed, not memory.
6. **Nyström-SHAP pre-print (2023)** – no public code; 32 GB RAM.
7. **Gap**: No open-source tool gives O(nk) memory & model-agnostic Shapley on ≤8 GB.