# Metrics

> **Note:** To render math equations, enable `pymdownx.arithmatex` in your `mkdocs.yml` and include MathJax. See the user guide for details.

mmm-eval provides a suite of metrics to evaluate MMM performance. This guide explains each metric and how to interpret the results.

Note that these metrics do not claim to be entirely comprehensive, but instead aim to provide an overall view
of MMM performance across several key dimensions.

## Overview

mmm-eval calculates several key metrics across different validation tests:

### Accuracy Metrics

- **MAPE (Mean Absolute Percentage Error)**: Average percentage error between predictions and actual values
- **SMAPE (Symmetric Mean Absolute Percentage Error)**: Symmetric version of MAPE that treats over and underestimation equally
- **R-squared**: Proportion of variance explained by the model

### Stability Metrics

- **Parameter Change**: Percentage change in model parameters
- **Channel Stability**: Stability of media channel coefficients
- **Intercept Stability**: Stability of baseline parameters

## Metric Definitions

### MAPE (Mean Absolute Percentage Error)

```python
MAPE = (1/n) * Σ |(y_i - ŷ_i) / y_i|
```

**Interpretation**:
- **Lower is better**: 0% = perfect predictions
- **Scale**: Expressed as a percentage, e.g. 15.0 rather than 0.15

### SMAPE (Symmetric Mean Absolute Percentage Error)

```python
SMAPE = 100 * (2 * |y_i - ŷ_i|) / (|y_i| + |ŷ_i|)
```

**Interpretation**:
- **Lower is better**: 0% = perfect predictions
- **Scale**: Expressed as a percentage, e.g. 15.0 rather than 0.15
- **Symmetric**: Treats over and underestimation equally (unlike MAPE)
- **Robust**: Less sensitive to extreme values and zero actual values

**Advantages over MAPE**:
- **Symmetry**: 10% overestimation and 10% underestimation give the same SMAPE value
- **Zero handling**: Better handling of zero or near-zero actual values
- **Bounded**: Upper bound of 200% vs. unbounded MAPE

### R-squared (Coefficient of Determination)

```python
R² = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)
```

**Interpretation**:
- **Range**: 0 to 1 (higher is better)
- **Scale**: 1 = perfect fit, 0 = no predictive power
- **Benchmark**: > 0.8 is generally good

## Test-Specific Metrics

### Holdout Accuracy Test Metrics

Metrics calculated on out-of-sample predictions using train/test splits.

- **MAPE**: Overall prediction accuracy
- **SMAPE**: Symmetric prediction accuracy
- **R-squared**: Model fit quality

### In-Sample Accuracy Test Metrics

Metrics calculated on in-sample predictions using the full dataset.

- **MAPE**: Model fit accuracy
- **SMAPE**: Symmetric model fit accuracy
- **R-squared**: Model fit quality

### Cross-Validation Metrics

- **Mean MAPE**: Average out-of-sample accuracy
- **Std MAPE**: Consistency of accuracy across folds
- **Mean SMAPE**: Average out-of-sample symmetric accuracy
- **Std SMAPE**: Consistency of symmetric accuracy across folds
- **Mean R-squared**: Average out-of-sample fit
- **Std R-squared**: Consistency of fit across folds

### Refresh Stability Metrics

- **Mean Percentage Change**: Average parameter change
- **Std Percentage Change**: Consistency of parameter changes
- **Channel-specific Stability**: Stability per media channel

### Perturbation Metrics

- **Percentage Change**: Change in ROI estimates when input data is perturbed
- **Channel-specific Sensitivity**: Sensitivity of each media channel to data perturbations
- **Model Robustness**: Overall model stability to input noise

### Placebo Metrics

- **Shuffled Channel ROI**: Estimated ROI for the spurious (shuffled) channel
- **Shuffled Channel Name**: Name of the channel that was shuffled for the test
- **Falsifiability Assessment**: Whether the model correctly identifies spurious correlations

## Interpreting Results

### Good Performance Indicators

- **MAPE < 15%**: Good prediction accuracy
- **SMAPE < 15%**: Good symmetric prediction accuracy
- **R-squared > 0.8**: Strong model fit
- **Low perturbation sensitivity**: Robust to input noise
- **Low placebo ROI (≤ -50%)**: Correctly identifies spurious correlations

## Thresholds and Benchmarks

### Rough Benchmarks

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| MAPE | < 5% | 5-10% | 10-15% | > 15% |
| SMAPE | < 5% | 5-10% | 10-15% | > 15% |
| R-squared | > 0.9 | 0.8-0.9 | 0.6-0.8 | < 0.6 |
| Parameter Change | < 5% | 5-10% | 10-20% | > 20% |
| Perturbation Change | < 5% | 5-10% | 10-15% | > 15% |
| Placebo ROI | ≤ -50% | -50% to -25% | -25% to 0% | > 0% |

## Customizing Metrics

### Modifying Thresholds

If you'd like to modify the test pass/fail thresholds, you can fork the branch and
modify the thresholds in `mmm_eval/metrics/threshold_constants.py`.

### Adding Custom Metrics

To add custom metrics, extend the metrics module:

```python
from mmm_eval.metrics import BaseMetric

class CustomMetric(BaseMetric):
    def calculate(self, y_true, y_pred):
        # Your custom calculation
        return custom_value
```

## Best Practices

### Metric Selection

- **Start with MAPE**: Most intuitive for business users
- **Include SMAPE**: More robust alternative to MAPE for symmetric evaluation
- **Include R-squared**: Technical measure of fit quality
- **Monitor stability**: Critical for production models
- **Track performance**: Important for scalability

### Result Analysis

- **Compare across frameworks**: Use same metrics for fair comparison
- **Track over time**: Monitor performance as data grows
- **Set business thresholds**: Align with business requirements
- **Document decisions**: Record metric choices and rationale

## Troubleshooting

### Common Issues

1. **Extreme MAPE values**: Check for zero or near-zero actual values
2. **High SMAPE values**: Check for symmetric errors and zero handling
3. **Negative R-squared**: Model performs worse than baseline
4. **Inconsistent metrics**: Verify data preprocessing
5. **Missing metrics**: Check test configuration

### Getting Help

- Review [Tests](../user-guide/tests.md) for metric context
- Check [Configuration](../getting-started/configuration.md) for settings
- Join [Discussions](https://github.com/mutinex/mmm-eval/discussions) for support 