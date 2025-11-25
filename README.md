# Q-Cluster Error Mitigation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-purple.svg)](https://qiskit.org/)

## 📖 Overview
This project implements an unsupervised machine learning approach, **Q-Cluster**, to mitigate readout errors in quantum computing. 

Quantum readout errors (measurement errors) are a significant noise source in NISQ (Noisy Intermediate-Scale Quantum) devices. This project utilizes a clustering algorithm based on **Hamming Distance** and **Qubit-wise Majority Voting (QMV)** to recover the ideal probability distribution from noisy measurement results without requiring additional quantum resources.

## 🚀 Key Features
- **Custom Noise Modeling**: Simulates realistic quantum noise, specifically focusing on asymmetric readout errors using `Qiskit Aer`.
- **Q-Cluster Algorithm**:
  - **Metric**: Hamming Distance for bitstring similarity.
  - **Update Rule**: Qubit-wise Majority Voting (QMV) to determine cluster centroids.
- **Visualization**:
  - Comparative histograms (Ideal vs. Noisy vs. Mitigated).
  - 2D PCA visualization of bitstring clusters in the latent space.

## 📂 Project Structure
```text
Q-Cluster-Mitigation/
├── src/
│   ├── __init__.py
│   ├── noise_models.py     # Custom noise model builder
│   ├── qcluster.py         # Q-Cluster algorithm implementation
│   └── visualization.py    # Plotting & PCA visualization tools
├── main.py                 # Main execution script
├── demo.ipynb              # Interactive Jupyter Notebook demo
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

