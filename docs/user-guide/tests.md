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
- **RMSE (Root Mean Square Error)**: Standard deviation of prediction errors
- **R-squared**: Proportion of variance explained by the model
- **MAE (Mean Absolute Error)**: Average absolute prediction error

### Interpretation

- **Lower MAPE/RMSE/MAE**: Better model performance
- **Higher R-squared**: Better model fit (0-1 scale)
- **Industry benchmarks**: MAPE < 20% is generally good

### Example Results

```json
{
  "accuracy": {
    "mape": 0.15,
    "rmse": 125.5,
    "r_squared": 0.85,
    "mae": 98.2
  }
}
```

## Cross-Validation Tests

Cross-validation tests assess how well the model generalizes to unseen data.

### Process

1. **Time Series Split**: Data is split chronologically
2. **Rolling Window**: Model is trained on expanding windows
3. **Out-of-Sample Prediction**: Predictions made on held-out data
4. **Performance Metrics**: Calculated on out-of-sample predictions

### Metrics

- **MAPE**: Out-of-sample prediction accuracy
- **RMSE**: Out-of-sample error magnitude
- **R-squared**: Out-of-sample explanatory power
- **MAE**: Out-of-sample absolute error

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
benjammmin --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/
```

### Specific Tests

```bash
benjammmin --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/ --test-names accuracy cross_validation
```

### Available Test Names

- `accuracy`: Accuracy tests only
- `cross_validation`: Cross-validation tests only
- `refresh_stability`: Refresh stability tests only
- `performance`: Performance tests only

## Test Configuration

### Accuracy Test Settings

```json
{
  "accuracy": {
    "train_test_split": 0.8,
    "random_seed": 42
  }
}
```

### Cross-Validation Settings

```json
{
  "cross_validation": {
    "n_splits": 5,
    "test_size": 0.2,
    "gap": 0
  }
}
```

### Refresh Stability Settings

```json
{
  "refresh_stability": {
    "baseline_periods": 52,
    "update_frequency": 4,
    "max_updates": 12
  }
}
```

## Interpreting Results

### Good Model Indicators

- **Accuracy**: MAPE < 20%, R-squared > 0.8
- **Cross-Validation**: Out-of-sample MAPE similar to in-sample
- **Stability**: Parameter changes < 10%
- **Performance**: Reasonable training times

### Warning Signs

- **Overfitting**: Much better in-sample than out-of-sample performance
- **Instability**: Large parameter changes with new data
- **Poor Performance**: High MAPE or low R-squared
- **Slow Training**: Excessive computation time

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

- Check [Configuration](getting-started/configuration.md) for test settings
- Review [Examples](examples/basic-usage.md) for similar cases
- Join [Discussions](https://github.com/Mutiny-Group/mmm-eval/discussions) for support 