# Tests

mmm-eval provides a comprehensive suite of validation tests to evaluate MMM performance. This guide explains each test and how to interpret the results.

## Overview

mmm-eval includes four main types of validation tests:

1. **Accuracy Test** - Predictive accuracy on unseen data
2. **Cross-Validation Test** - Predictive accuracy on k-folds of unseen data
3. **Refresh Stability Test** - Stability of marketing attribution when additional data is added
4. **Perturbation Test** - Stability of marketing attribution when noise is added to input data

## Accuracy Test

### Purpose

The accuracy test evaluates how well the model predicts on unseen data.

### Methodology

1. **Data Split**: Divides data into training and test sets
2. **Model Training**: Trains the model on the training set
3. **Prediction**: Makes predictions on the test set
4. **Evaluation**: Calculates accuracy metrics

### Interpretation

- **Good accuracy**: Low MAPE/RMSE, high R²
- **Poor accuracy**: High MAPE/RMSE, low R²
- **Overfitting**: Good training performance, poor test performance

## Cross-Validation Test

### Purpose

The cross-validation test assesses the *consistency* of model accuracy across multiple data splits.

### Methodology

1. **K-Fold Split**: Divides data into k contiguous folds (to preserve time series structure)
2. **Iterative Training**: Trains k models, each using k-1 folds
3. **Performance Assessment**: Evaluates each model on the held-out fold
4. **Stability Analysis**: Measures variation in accuracy metrics across folds

### Interpretation

- **Stable model**: Low standard deviation across folds
- **Unstable model**: High standard deviation across folds
- **Overfitting**: High variation suggests poor generalization

## Refresh Stability Test

### Purpose

The refresh stability test evaluates how stable marketing attribution is when new data is added (refreshed).

### Methodology

1. **Progressive Training**: Trains models on increasing proportions of data
2. **Performance Tracking**: Measures performance at each refresh point
3. **Stability Assessment**: Analyzes variation in performance over time

### Interpretation

- **Stable model**: Consistent attribution across refresh periods
- **Unstable model**: Attribution varies significantly over time
- **Improving model**: Attribution improves with more data

## Perturbation Test

### Purpose

The perturbation test evaluates how sensitive the model is to small changes in the data.

### Methodology

1. **Train a Model**: Train a model on the original data.
2. **Data Perturbation**: Adds small random noise to marketing input data
3. **Model Retraining**: Retrains model on perturbed data
4. **Performance Comparison**: Compares marketing attribution between perturbed and non-perturbed models


### Interpretation

- **Robust model**: Performance changes little with perturbations
- **Sensitive model**: Performance degrades significantly with perturbations
- **Overfitting**: High sensitivity suggests poor generalization

## Test Selection

### When to Use Each Test

| Test | Best For | When to Use |
|------|----------|-------------|
| Accuracy | Basic evaluation | Initial model assessment |
| Cross-Validation | Stability assessment | Model comparison |
| Refresh Stability | Temporal analysis | Long-term planning |
| Perturbation | Robustness evaluation | Production readiness |

### Recommended Test Combinations
All tests are run by default, which is the recommendation. However, users can specify a subset of tests to run

CLI
```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/ --test-names accuracy cross_validation
```

Python
```python
results = run_evaluation(config=config, data=data, framework="pymc-marketing", test_names = ("accuracy","cross_validation"))
```

## Interpreting Test Results

Each test answers a distinct question:

* **Accuracy Test**: "How well does my model predict on unseen data?"
* **Cross-Validation**: "How *consistent* are my model's predictions across different splits of unseen data?"
* **Refresh Stability**: "How much does marketing attribution change when I add new data to my model?"
* **Perturbation**: "How sensitive is my model is to noise in the marketing inputs?"

For each test, we compute multiple metrics to give as much insight into the test result as possible: 

* **MAPE (Mean Absolute Percentage Error)**  
  `MAPE = (100 / n) * Σ |(y_i - ŷ_i) / y_i|`

* **RMSE (Root Mean Square Error)**  
  `RMSE = sqrt((1 / n) * Σ (y_i - ŷ_i)^2)`

* **R-squared (Coefficient of Determination)**  
  `R² = 1 - (Σ (y_i - ŷ_i)^2) / (Σ (y_i - ȳ)^2)`

### Example Results

|     test_name     |                  metric_name                  | metric_value | metric_pass |
|-------------------|-----------------------------------------------|--------------|-------------|
| accuracy          | mape                                          | 0.121        | False       |
| accuracy          | r_squared                                     | -0.547       | False       |
| cross_validation  | mean_mape                                     | 0.084        | False       |
| cross_validation  | std_mape                                      | 0.058        | False       |
| cross_validation  | mean_r_squared                                | -7.141       | False       |
| cross_validation  | std_r_squared                                 | 9.686        | False       |
| refresh_stability | mean_percentage_change_for_each_channel:TV    | 0.021        | False       |
| refresh_stability | mean_percentage_change_for_each_channel:radio | 0.369        | False       |
| refresh_stability | std_percentage_change_for_each_channel:TV     | 0.021        | False       |
| refresh_stability | std_percentage_change_for_each_channel:radio  | 0.397        | False       |
| perturbation      | percentage_change_for_each_channel:TV         | 0.005        | False       |
| perturbation      | percentage_change_for_each_channel:radio      | 0.112        | False       |


## Changing the Thresholds
Default metric thresholds in `mmm_eval/metrics/threshold_constants.py` can be overwritten in-place to change the pass/fail cutoff for each metric.


## Next Steps

- Learn about [Metrics](metrics.md) to understand test outputs
- Check [Examples](examples/basic-usage.md) for practical test usage
- Review [Configuration](getting-started/configuration.md) for test customization 