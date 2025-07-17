# Tests

mmm-eval provides a comprehensive suite of validation tests to evaluate MMM performance. This guide explains each test and how to interpret the results.

## Overview

mmm-eval includes four main types of validation tests:

1. **Accuracy Tests**: Measure how well the model fits the data
2. **Cross-Validation Tests**: Assess model generalization
3. **Refresh Stability Tests**: Evaluate model stability over time
4. **Performance Tests**: Measure computational efficiency

## Accuracy Tests

Accuracy tests evaluate how well the model fits the training data.

### Metrics

- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **SMAPE (Symmetric Mean Absolute Percentage Error)**: Symmetric version of MAPE
- **R-squared**: Proportion of variance explained by the model

### Interpretation

- **Lower MAPE**: Better model performance
- **Lower SMAPE**: Better symmetric model performance
- **Higher R-squared**: Better model fit (0-1 scale)

## Cross-Validation Tests

Cross-validation tests assess how well the model generalizes to unseen data.

### Process

1. **Time Series Split**: Data is split chronologically
2. **Rolling Window**: Model is trained on expanding windows
3. **Out-of-Sample Prediction**: Predictions made on held-out data
4. **Performance Metrics**: Calculated on out-of-sample predictions

### Metrics

- **MAPE**: Out-of-sample prediction accuracy
- **SMAPE**: Out-of-sample symmetric prediction accuracy
- **R-squared**: Out-of-sample explanatory power

### Interpretation

- **Consistent performance**: Similar in-sample and out-of-sample metrics
- **Overfitting**: Much better in-sample than out-of-sample performance
- **Underfitting**: Poor performance on both in-sample and out-of-sample data

## Refresh Stability Tests

Refresh stability tests evaluate how model parameters change when new data is added.

### Process

1. **Baseline Model**: Train on initial dataset
2. **Incremental Updates**: Add new data periods
3. **Parameter Comparison**: Compare parameter estimates
4. **Stability Metrics**: Calculate change percentages

### Metrics

- **Mean Percentage Change**: Average change in parameter estimates
- **Channel Stability**: Stability of media channel parameters
- **Intercept Stability**: Stability of baseline parameters
- **Seasonality Stability**: Stability of seasonal components

### Interpretation

- **Low percentage changes**: Stable model parameters
- **High percentage changes**: Unstable model (may need more data)
- **Channel-specific stability**: Some channels more stable than others

## Performance Tests

Performance tests measure computational efficiency and resource usage.

### Metrics

- **Training Time**: Time to fit the model
- **Memory Usage**: Peak memory consumption
- **Prediction Time**: Time to generate predictions
- **Convergence**: Number of iterations to convergence

### Interpretation

- **Faster training**: More efficient model
- **Lower memory**: Better resource utilization
- **Faster predictions**: Better for real-time applications
- **Fewer iterations**: Better convergence properties

## Running Tests

### All Tests (Default)

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/
```

### Specific Tests

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/ --test-names accuracy cross_validation
```

### Available Test Names

- `accuracy`: Accuracy tests only
- `cross_validation`: Cross-validation tests only
- `refresh_stability`: Refresh stability tests only
- `performance`: Performance tests only

## Test Configuration

If you'd like to modify the test pass/fail thresholds, you can fork the branch and
modify the thresholds in `mmm_eval/metrics/threshold_constants.py`.

## Interpreting Results

### Good Model Indicators

- **Accuracy**: MAPE < 15%, SMAPE < 15%, R-squared > 0.8
- **Cross-Validation**: Out-of-sample MAPE/SMAPE similar to in-sample
- **Refresh Stability**: Parameter changes < 10%
- **Perturbation**: ROI changes < 5%

### Warning Signs

- **Poor Performance**: High MAPE/SMAPE or low R-squared
- **Unstable Model**: Large parameter changes
- **Overfitting**: In-sample vs out-of-sample performance gap
- **Data Issues**: Missing values or extreme outliers

## Best Practices

### Test Selection

- **Start with accuracy**: Always run accuracy tests first
- **Add cross-validation**: For generalization assessment
- **Include stability**: For production models
- **Monitor performance**: For computational constraints

### Result Analysis

- **Compare frameworks**: Run same tests on different frameworks
- **Track over time**: Monitor performance as data grows
- **Set thresholds**: Define acceptable performance levels
- **Document decisions**: Record test choices and rationale

## Troubleshooting

### Common Issues

1. **Slow tests**: Reduce data size or simplify model
2. **Memory errors**: Use smaller datasets or more efficient settings
3. **Convergence issues**: Check model configuration
4. **Inconsistent results**: Verify random seed settings

### Getting Help

- Check [Configuration](../getting-started/configuration.md) for test settings
- Review [Examples](../examples/basic-usage.md) for similar cases
- Join [Discussions](https://github.com/mutinex/mmm-eval/discussions) for support 