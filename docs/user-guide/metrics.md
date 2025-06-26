# Metrics

mmm-eval provides a comprehensive set of metrics to evaluate MMM performance. This guide explains each metric and how to interpret the results.

## Overview

mmm-eval calculates several key metrics across different validation tests:

- **Accuracy Metrics**: How well the model predicts on unseen data
- **Stability Metrics**: How consistent the model is over time
- **Robustness Metrics**: How sensitive the model is to data changes

## Accuracy Metrics

### MAPE (Mean Absolute Percentage Error)

**Formula**: MAPE = (100% / n) * Σ|(y_i - ŷ_i) / y_i|

**Interpretation**: 
- Lower values indicate better accuracy
- Expressed as a percentage
- Sensitive to scale of target variable

**Example**: MAPE of 15% means predictions are off by 15% on average

### RMSE (Root Mean Square Error)

**Formula**: RMSE = √(Σ(y_i - ŷ_i)² / n)

**Interpretation**:
- Lower values indicate better accuracy
- Expressed in same units as target variable
- Penalizes large errors more heavily than small ones

**Example**: RMSE of 100 means predictions deviate by 100 units on average

### R-squared (Coefficient of Determination)

**Formula**: R² = 1 - (Σ(y_i - ŷ_i)² / Σ(y_i - ȳ)²)

**Interpretation**:
- Range: 0 to 1 (or 0% to 100%)
- Higher values indicate better fit
- Represents proportion of variance explained by the model

**Example**: R² of 0.85 means the model explains 85% of the variance

### MAE (Mean Absolute Error)

**Formula**: MAE = Σ|y_i - ŷ_i| / n

**Interpretation**:
- Lower values indicate better accuracy
- Expressed in same units as target variable
- Less sensitive to outliers than RMSE

**Example**: MAE of 80 means predictions deviate by 80 units on average

### MSE (Mean Square Error)

**Formula**: MSE = Σ(y_i - ŷ_i)² / n

**Interpretation**:
- Lower values indicate better accuracy
- Expressed in squared units of target variable
- Heavily penalizes large errors

**Example**: MSE of 10,000 means average squared error is 10,000

## Stability Metrics

### Refresh Stability

Measures *consistency* of channel attribution when the model is trained on different time periods.

**Calculation**:
1. Train model on different proportions of data (e.g., 50%, 75%, 90%)
2. Calculate metrics for each refresh period
3. Measure average and standard deviation in metrics across periods

**Interpretation**:
- Lower variation indicates more stable model
- High variation suggests model is sensitive to training data

## Robustness Metrics

### Perturbation Sensitivity

Measures how sensitive the model is to small changes in the data.

**Calculation**:
1. Add small random perturbations to input data
2. Retrain model and calculate metrics
3. Measure change in performance

**Interpretation**:
- Lower sensitivity indicates more robust model
- High sensitivity suggests model may not generalize well

## Metric Comparison

### When to Use Each Metric

| Metric | Best For | Considerations |
|--------|----------|----------------|
| MAPE | Relative accuracy | Sensitive to scale, good for comparison |
| RMSE | Overall accuracy | Penalizes large errors heavily |
| R² | Model fit quality | May be misleading with non-linear relationships |
| MAE | Robust accuracy | Less sensitive to outliers |
| MSE | Mathematical optimization | Harder to interpret |

### Metric Ranges and Benchmarks

#### MAPE Benchmarks
- **Excellent**: < 10%
- **Good**: 10% - 20%
- **Fair**: 20% - 30%
- **Poor**: > 30%

#### R² Benchmarks
- **Excellent**: > 0.9
- **Good**: 0.8 - 0.9
- **Fair**: 0.6 - 0.8
- **Poor**: < 0.6

*Note: These benchmarks are general guidelines. Industry-specific benchmarks may vary.*

## Interpreting Results

### Single Model Evaluation

When evaluating a single model:

1. **Check accuracy metrics**: Are predictions reasonably accurate?
2. **Assess stability**: Is performance consistent across different scenarios?
3. **Evaluate robustness**: How sensitive is the model to data changes?

### Model Comparison

When comparing multiple models:

1. **Compare accuracy**: Which model has better predictive performance?
2. **Compare stability**: Which model is more consistent?
3. **Compare robustness**: Which model is less sensitive to changes?

## Best Practices

### Metric Selection

1. **Use multiple metrics**: Don't rely on a single metric
2. **Consider business context**: Choose metrics relevant to your goals
3. **Account for scale**: Use relative metrics for comparison across scales

### Result Interpretation

1. **Set realistic expectations**: MMM accuracy varies by industry
2. **Consider data quality**: Poor data leads to poor metrics
3. **Validate assumptions**: Ensure metrics align with business needs

### Continuous Monitoring

1. **Track metrics over time**: Monitor performance degradation
2. **Set up alerts**: Flag when metrics fall below thresholds
3. **Regular re-evaluation**: Periodically reassess model performance

## Next Steps

- Learn about [Tests](tests.md) to understand how metrics are calculated
- Check [Examples](examples/basic-usage.md) for practical metric interpretation
- Review [Configuration](getting-started/configuration.md) for customizing metric calculation 