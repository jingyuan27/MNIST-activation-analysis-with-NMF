# Experiment 05.1: Quantitative Analysis of Neuron Structure

## 1. Per-neuron Sparsity

The ReLU network exhibits heterogeneous sparsity across its 128 hidden neurons. The per-neuron zero activation ratios range approximately from 0 to 0.8, with most neurons concentrated between 0.2 and 0.6.

The global zero activation ratio is approximately 38.43%.

This suggests that sparsity is not uniformly distributed across hidden neurons.

## 2. Tanh Saturation

The Tanh network exhibits substantial activation saturation.

Using thresholds of 0.95 and 0.99, the per-neuron saturation ratios show considerable variation. The saturation ratio at the stricter threshold is consistently lower.

This indicates heterogeneous saturation behavior among hidden neurons.

## 3. Class Selectivity

The eta-squared distributions of ReLU and Tanh overlap considerably. The Tanh distribution appears slightly shifted toward higher values.

This suggests that nonnegative activations and class selectivity are distinct properties. Greater sparsity does not necessarily imply stronger class selectivity.

## 4. Preliminary Conclusions

The experiment reveals three distinct structural properties:

- ReLU produces sparse, nonnegative representations.
- Tanh produces signed representations with substantial saturation.
- Both networks exhibit neuron-level class-dependent activation patterns.

These observations motivate further investigation into whether the nonnegative structure of ReLU activations facilitates meaningful NMF decompositions.