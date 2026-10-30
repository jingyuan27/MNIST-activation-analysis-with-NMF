
# MNIST Activation Analysis with NMF

**Understanding Neural Network Representations through Non-negative Matrix Factorization**

## 1. Overview

This project investigates the relationship between
neural network activation functions and non-negative
matrix factorization (NMF).

Using the MNIST handwritten digit dataset, we train
fully connected neural networks with ReLU and Tanh
activation functions, extract hidden-layer activation
matrices, and analyze their numerical structures.

We focus on whether non-negative activations tend to
produce sparse, additive, and interpretable feature
representations.

## 2. Research Questions

- How do ReLU and Tanh affect hidden-layer activation
  distributions?
- Are ReLU activations more likely to produce sparse
  representations?
- Can NMF extract meaningful latent features from
  ReLU activation matrices?
- Do the extracted features correspond to recognizable
  patterns or basic structures in handwritten digits?

## 3. Methodology

The project consists of four main steps:

1. Train neural networks on MNIST.
2. Extract hidden-layer activation matrices.
3. Apply NMF to non-negative activation matrices.
4. Visualize and compare learned representations.

For a hidden-layer activation matrix A, we investigate
the factorization:

    A ≈ WH

where W represents sample-level latent coefficients
and H represents latent components in neuron space.

## 4. Repository Structure

- `src/`: Core implementation
- `scripts/`: Experiment execution scripts
- `notebooks/`: Exploratory experiments
- `configs/`: Experiment configurations
- `tests/`: Unit tests
- `docs/`: Learning notes and mathematical background
- `reports/`: Experimental results and figures

## 5. Environment

Python 3.10+

Main dependencies:

- NumPy
- PyTorch
- Torchvision
- Scikit-learn
- Matplotlib
- Pandas

## 6. Experiments

### Baseline
- MNIST classification
- ReLU vs. Tanh

### Activation Analysis
- Activation distributions
- Activation sparsity
- Neuron selectivity

### Matrix Factorization
- NMF decomposition
- Reconstruction error
- Latent feature visualization

### Comparative Analysis
- ReLU vs. Tanh
- Different NMF component numbers
- Repeated experiments with multiple random seeds

## 7. Reproducibility

All experiments are associated with explicit
configuration files and random seeds.

MNIST data and large intermediate artifacts are not
included in this repository.

## 8. Results

Experimental results and visualizations will be
documented in `reports/`.

## 9. References

- Saito, K. Deep Learning from Scratch:
  Theory and Implementation.
- Scikit-learn NMF Documentation.
- PyTorch Documentation.

## 10. Limitations

Non-negative activations do not necessarily imply
interpretable or localized features.

NMF is used as an exploratory factorization method.
The project does not assume a strict mathematical
equivalence between ReLU networks and NMF.

