# NMPPI
**Multi-scale Relationship Structure with Effective Pre-training for Protein–Protein Interactions**

This repository provides the official implementation of **NMPPI**, a motif-aware and
multi-scale framework for protein–protein interaction (PPI) prediction. The proposed
method integrates effective pre-training with multi-scale relational structure modeling
to address the challenges of long-range dependency modeling and degree heterogeneity
in large-scale and sparsely connected PPI networks.

---

## 🔧 Requirements

The code is implemented in Python. The following dependencies are required:

- Python >= 3.8  
- PyTorch >= 1.12  
- torch-geometric  
- numpy  
- scipy  
- scikit-learn  
- networkx  

You can install the required packages using:

```bash
pip install -r requirements.txt
