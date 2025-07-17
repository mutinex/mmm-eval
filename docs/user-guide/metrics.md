# Metrics

mmm-eval provides comprehensive metrics for evaluating MMM performance across different validation approaches. This guide explains each metric and how to interpret the results.

## Overview

mmm-eval calculates metrics for different types of validation tests:

1. **Accuracy Metrics**: Model fit and prediction accuracy
2. **Cross-Validation Metrics**: Generalization performance
3. **Refresh Stability Metrics**: Model stability over time
4. **Perturbation Metrics**: Model robustness

## Accuracy Metrics

### Holdout Accuracy Test Metrics

Metrics calculated on out-of-sample predictions using train/test splits.

#### Primary Metrics

- **MAPE**: Overall prediction accuracy
- **SMAPE**: Symmetric prediction accuracy
- **R-squared**: Proportion of variance explained

#### Interpretation

- **MAPE < 15%**: Good prediction accuracy
- **SMAPE < 15%**: Good symmetric prediction accuracy
- **R-squared > 0.8**: Good explanatory power

### In-Sample Accuracy Test Metrics

Metrics calculated on in-sample predictions using the full dataset.

#### Primary Metrics

- **MAPE**: Model fit accuracy
- **SMAPE**: Symmetric model fit accuracy
- **R-squared**: Proportion of variance explained

#### Interpretation

- **MAPE < 10%**: Excellent model fit
- **SMAPE < 10%**: Excellent symmetric model fit
- **R-squared > 0.9**: Very good explanatory power
- **Comparison with holdout**: Much better in-sample than holdout indicates overfitting

## Cross-Validation Metrics

### Cross-Validation Test Metrics

Metrics calculated across multiple time-series folds.

#### Primary Metrics

- **Mean MAPE**: Average out-of-sample accuracy
- **Std MAPE**: Consistency of accuracy across folds
- **Mean SMAPE**: Average out-of-sample symmetric accuracy
- **Std SMAPE**: Consistency of symmetric accuracy across folds
- **Mean R-squared**: Average explanatory power

#### Interpretation

- **Low mean values**: Good average performance
- **Low std values**: Consistent performance across folds
- **High R-squared**: Good explanatory power

## Refresh Stability Metrics

### Refresh Stability Test Metrics

Metrics measuring parameter stability when new data is added.

#### Primary Metrics

- **Mean Percentage Change**: Average change in parameter estimates
- **Std Percentage Change**: Consistency of parameter changes

#### Interpretation

- **Mean < 10%**: Stable model parameters
- **Std < 5%**: Consistent parameter stability
- **High values**: Unstable model (may need more data)

## Perturbation Metrics

### Perturbation Test Metrics

Metrics measuring model robustness to data perturbations.

#### Primary Metrics

- **Percentage Change**: Change in ROI estimates after perturbation

#### Interpretation

- **Change < 5%**: Robust model
- **Change > 10%**: Sensitive model
- **Channel-specific**: Some channels more robust than others

## Metric Thresholds

### Good Model Indicators

- **Holdout Accuracy**: MAPE < 15%, SMAPE < 15%, R-squared > 0.8
- **In-Sample Accuracy**: MAPE < 10%, SMAPE < 10%, R-squared > 0.9
- **Cross-Validation**: Mean MAPE < 15%, Std MAPE < 5%
- **Refresh Stability**: Mean change < 10%, Std change < 5%
- **Perturbation**: ROI change < 5%

### Warning Signs

- **Poor Performance**: High MAPE/SMAPE or low R-squared
- **Overfitting**: Much better in-sample than holdout performance
- **Unstable Model**: Large parameter changes
- **Inconsistent Performance**: High standard deviations

## Metric Calculations

### MAPE (Mean Absolute Percentage Error)

```python
MAPE = (1/n) * Σ|(actual - predicted) / actual| * 100
```

### SMAPE (Symmetric Mean Absolute Percentage Error)

```python
SMAPE = (2/n) * Σ|actual - predicted| / (|actual| + |predicted|) * 100
```

### R-squared

```python
R² = 1 - (SS_res / SS_tot)
```

Where:
- SS_res = Sum of squared residuals
- SS_tot = Total sum of squares

## Best Practices

### Metric Selection

- **Start with holdout accuracy**: Always evaluate out-of-sample performance
- **Add in-sample accuracy**: To assess model fit and identify overfitting
- **Include cross-validation**: For robust generalization assessment
- **Monitor stability**: For production model reliability
- **Test robustness**: For model sensitivity analysis

### Result Analysis

- **Compare frameworks**: Run same metrics on different frameworks
- **Track over time**: Monitor metrics as data grows
- **Set thresholds**: Define acceptable performance levels
- **Document decisions**: Record metric choices and rationale

## Troubleshooting

### Common Issues

1. **High MAPE**: Check data quality and model specification
2. **Low R-squared**: Consider additional features or model complexity
3. **Unstable parameters**: May need more data or regularization
4. **Overfitting**: Reduce model complexity or add regularization

### Getting Help

- Check [Configuration](../getting-started/configuration.md) for metric settings
- Review [Examples](../examples/basic-usage.md) for similar cases
- Join [Discussions](https://github.com/mutinex/mmm-eval/discussions) for support 