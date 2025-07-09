# Metrics

> **Note:** To render math equations, enable `pymdownx.arithmatex` in your `mkdocs.yml` and include MathJax. See the user guide for details.

BenjaMMMin provides a comprehensive set of metrics to evaluate MMM performance. This guide explains each metric and how to interpret the results.

## Overview

BenjaMMMin calculates several key metrics across different validation tests:

- **Accuracy Metrics**: How well the model predicts on unseen data
- **Stability Metrics**: How consistent the model is over time
- **Robustness Metrics**: How sensitive the model is to data changes

## Accuracy Metrics

### MAPE (Mean Absolute Percentage Error)

**Formula:**

$$
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

**Interpretation**:

- Lower values indicate better accuracy
- Expressed as a proportion (0 to 1, instead of 0 to 100)
- Sensitive to scale of target variable

**Example**: MAPE of 15% means predictions are off by 15% on average

### R-squared (Coefficient of Determination)

**Formula:**

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

**Interpretation**:

- Range: 0 to 1 (or 0% to 100%)
- Higher values indicate better fit
- Represents proportion of variance explained by the model

**Example**: R² of 0.85 means the model explains 85% of the variance

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