# Metrics

> **Note:** To render math equations, enable `pymdownx.arithmatex` in your `mkdocs.yml` and include MathJax. See the user guide for details.

mmm-eval provides a comprehensive set of metrics to evaluate MMM performance. This guide explains each metric and how to interpret the results.

## Overview

mmm-eval provides a comprehensive set of metrics to evaluate MMM performance. This guide explains each metric and how to interpret the results.

## Available Metrics

mmm-eval calculates several key metrics across different validation tests:

### Accuracy Metrics

- **MAPE (Mean Absolute Percentage Error)**: Average percentage error between predictions and actual values
- **SMAPE (Symmetric Mean Absolute Percentage Error)**: Symmetric version of MAPE that treats over and underestimation equally
- **R-squared**: Proportion of variance explained by the model

### Stability Metrics

- **Parameter Change**: Percentage change in model parameters
- **Channel Stability**: Stability of media channel coefficients
- **Intercept Stability**: Stability of baseline parameters

### Performance Metrics

- **Training Time**: Time required to fit the model
- **Memory Usage**: Peak memory consumption during training
- **Prediction Time**: Time to generate predictions
- **Convergence**: Number of iterations to reach convergence

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
- **Range**: 0% to 200% (200% when actual = 0 and predicted ≠ 0)

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

### Accuracy Test Metrics

- **MAPE**: Overall prediction accuracy
- **SMAPE**: Symmetric prediction accuracy
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

### Performance Metrics

- **Training Time**: Time required to fit the model
- **Memory Usage**: Peak memory consumption during training
- **Prediction Time**: Time to generate predictions
- **Convergence**: Number of iterations to reach convergence

## Interpreting Results

### Good Performance Indicators

- **MAPE < 15%**: Good prediction accuracy
- **SMAPE < 15%**: Good symmetric prediction accuracy
- **R-squared > 0.8**: Strong model fit
- **Low parameter changes**: Stable model
- **Reasonable training time**: Efficient computation

### Warning Signs

- **MAPE > 30%**: Poor prediction accuracy
- **SMAPE > 30%**: Poor symmetric prediction accuracy
- **R-squared < 0.5**: Weak model fit
- **High parameter changes**: Unstable model
- **Excessive training time**: Computational issues

## Thresholds and Benchmarks

### Industry Benchmarks

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| MAPE | < 5% | 5-10% | 10-15% | > 15% |
| SMAPE | < 5% | 5-10% | 10-15% | > 15% |
| R-squared | > 0.9 | 0.8-0.9 | 0.6-0.8 | < 0.6 |
| Parameter Change | < 5% | 5-10% | 10-20% | > 20% |

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