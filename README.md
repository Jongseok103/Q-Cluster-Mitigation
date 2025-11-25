# Q-Cluster Error Mitigation Implementation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-purple.svg)](https://qiskit.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.10801-B31B1B.svg)](https://arxiv.org/abs/2504.10801v1)

## 📜 Paper Reference
This repository contains a Python implementation of the **Q-Cluster** algorithm, as proposed in the research paper:

> **"Q-Cluster: Quantum Error Mitigation Through Noise-Aware Unsupervised Learning"** > *Hrushikesh Pramod Patil, Dror Baron, and Huiyang Zhou (NC State University)* > arXiv preprint arXiv:2504.10801 (2025).

## 📖 Overview
Quantum error mitigation (QEM) is critical in reducing the impact of noise in the pre-fault-tolerant era. This project implements **Q-Cluster**, a novel QEM approach that reshapes the measured bit-string distribution using unsupervised learning (clustering).

Based on the paper's methodology, this implementation focuses on:
1.  **Clustering**: Grouping noisy measurement results (bit-strings) based on **Hamming distance** to identify dominant structures.
2.  **Centroid Calculation**: Using **Qubit-wise Majority Vote (QMV)** to determine the noise-free centroid of each cluster.
3.  **Distribution Reshaping**: Adjusting the noisy distribution using Bayesian inference and bit-flip error rates to reverse noise effects.

## 🚀 Key Features
- **Custom Noise Modeling**: Simulates realistic quantum noise, specifically focusing on asymmetric readout errors as described in the study.
- **Q-Cluster Algorithm Implementation**:
  - Implements the iterative clustering approach to discover $K$ dominant bit-strings.
  - Filters outliers based on variance thresholds derived from the bit-flip noise model.
- **Visualization**:
  - Comparative histograms (Ideal vs. Noisy vs. Mitigated).
  - 2D PCA visualization of bitstring clusters in the latent space.

## 📂 Project Structure
```text
Q-Cluster-Mitigation/
├── src/
│   ├── __init__.py
│   ├── noise_models.py     # Custom noise model builder
│   ├── qcluster.py         # Q-Cluster algorithm core logic
│   └── visualization.py    # Plotting & PCA visualization tools
├── main.py                 # Main execution script
├── demo.ipynb              # Interactive Jupyter Notebook demo
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
````

## 💻 Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/Q-Cluster-Mitigation.git](https://github.com/your-username/Q-Cluster-Mitigation.git)
cd Q-Cluster-Mitigation
pip install -r requirements.txt
```

## 🔧 Usage

### 1\. Run the Simulation

You can run the full simulation pipeline, which includes noise modeling, ansatz execution, and mitigation:

```bash
python main.py
```

### 2\. Interactive Demo

Open `demo.ipynb` to explore the algorithm step-by-step with visualizations:

```bash
jupyter notebook demo.ipynb
```

## 📊 Methodology Summary

According to the research, the Q-Cluster workflow implemented here follows these steps:

1.  **Assumption**: Erroneous bit-strings tend to "cluster" around ideal ones in Hamming space under a bit-flip noise model.
2.  **Clustering**: The algorithm identifies clusters using K-Means with Hamming distance.
3.  **Refinement**: Outliers are removed if their distance from the centroid exceeds a threshold derived from the error rate variance ($HD > 2 \times Var$).
4.  **Redistribution**: The probability of non-centroid bit-strings is reduced, and the probability of centroids is boosted to recover the ideal distribution.

## 🔗 Citation

If you use this code or the concepts in your research, please cite the original paper:

```bibtex
@misc{patil2025qcluster,
      title={Q-Cluster: Quantum Error Mitigation Through Noise-Aware Unsupervised Learning}, 
      author={Hrushikesh Pramod Patil and Dror Baron and Huiyang Zhou},
      year={2025},
      eprint={2504.10801},
      archivePrefix={arXiv},
      primaryClass={quant-ph}
}
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome\!

## 📝 License

This project is licensed under the MIT License.

```
```
