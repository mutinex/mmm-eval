# Tests

mmm-eval provides a comprehensive suite of validation tests to evaluate MMM performance. This guide explains each test and how to interpret the results.

## Overview

mmm-eval includes four main types of validation tests:

1. **Accuracy Test** - Evaluates predictive performance
2. **Cross-Validation Test** - Assesses model stability
3. **Refresh Stability Test** - Tests temporal consistency
4. **Perturbation Test** - Evaluates robustness

## Accuracy Test

### Purpose

The accuracy test evaluates how well the model predicts on unseen data.

### Methodology

1. **Data Split**: Divides data into training and test sets
2. **Model Training**: Trains the model on the training set
3. **Prediction**: Makes predictions on the test set
4. **Evaluation**: Calculates accuracy metrics

### Configuration

```json
{
  "tests": {
    "accuracy": {
      "train_test_split": 0.8,
      "random_state": 42,
      "metrics": ["mape", "rmse", "r2", "mae"]
    }
  }
}
```

### Interpretation

- **Good accuracy**: Low MAPE/RMSE, high R²
- **Poor accuracy**: High MAPE/RMSE, low R²
- **Overfitting**: Good training performance, poor test performance

## Cross-Validation Test

### Purpose

The cross-validation test assesses model stability across different data splits.

### Methodology

1. **K-Fold Split**: Divides data into k equal parts
2. **Iterative Training**: Trains k models, each using k-1 folds
3. **Performance Assessment**: Evaluates each model on the held-out fold
4. **Stability Analysis**: Measures variation in performance across folds

### Configuration

```json
{
  "tests": {
    "cross_validation": {
      "folds": 5,
      "random_state": 42,
      "metrics": ["mape", "rmse", "r2", "mae"]
    }
  }
}
```

### Interpretation

- **Stable model**: Low standard deviation across folds
- **Unstable model**: High standard deviation across folds
- **Overfitting**: High variation suggests poor generalization

## Refresh Stability Test

### Purpose

The refresh stability test evaluates how consistent model performance is over time.

### Methodology

1. **Progressive Training**: Trains models on increasing proportions of data
2. **Performance Tracking**: Measures performance at each refresh point
3. **Stability Assessment**: Analyzes variation in performance over time

### Configuration

```json
{
  "tests": {
    "refresh_stability": {
      "refresh_periods": [0.5, 0.75, 0.9],
      "metrics": ["mape", "rmse", "r2", "mae"]
    }
  }
}
```

### Interpretation

- **Stable model**: Consistent performance across refresh periods
- **Unstable model**: Performance varies significantly over time
- **Improving model**: Performance improves with more data

## Perturbation Test

### Purpose

The perturbation test evaluates how sensitive the model is to small changes in the data.

### Methodology

1. **Data Perturbation**: Adds small random noise to input data
2. **Model Retraining**: Retrains model on perturbed data
3. **Performance Comparison**: Compares performance with original model
4. **Sensitivity Analysis**: Measures change in performance

### Configuration

```json
{
  "tests": {
    "perturbation": {
      "perturbation_levels": [0.05, 0.1, 0.15],
      "metrics": ["mape", "rmse", "r2", "mae"]
    }
  }
}
```

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

#### Basic Evaluation
```bash
mmm-eval --test-names accuracy
```

#### Comprehensive Evaluation
```bash
mmm-eval --test-names accuracy,cross_validation,refresh_stability,perturbation
```

#### Production Readiness
```bash
mmm-eval --test-names cross_validation,perturbation
```

## Interpreting Test Results

### Single Test Results

Each test provides specific insights:

- **Accuracy Test**: Overall predictive performance
- **Cross-Validation**: Model stability and generalization
- **Refresh Stability**: Temporal consistency
- **Perturbation**: Robustness to data changes

### Combined Results

When all tests are run together:

1. **Check accuracy**: Is the model predicting well?
2. **Assess stability**: Is performance consistent?
3. **Evaluate robustness**: Is the model reliable?

### Example Results

```json
{
  "accuracy": {
    "mape": 12.5,
    "rmse": 150.2,
    "r2": 0.87
  },
  "cross_validation": {
    "mape_mean": 13.2,
    "mape_std": 1.8,
    "r2_mean": 0.85,
    "r2_std": 0.03
  },
  "refresh_stability": {
    "mape_variation": 2.1,
    "r2_variation": 0.05
  },
  "perturbation": {
    "mape_sensitivity": 0.15,
    "r2_sensitivity": 0.02
  }
}
```

**Interpretation**:
- Good accuracy (MAPE 12.5%, R² 0.87)
- Stable cross-validation (low std dev)
- Good refresh stability (low variation)
- Robust to perturbations (low sensitivity)

## Test Configuration

### Customizing Test Parameters

You can customize test parameters through configuration:

```json
{
  "tests": {
    "accuracy": {
      "train_test_split": 0.8,
      "random_state": 42
    },
    "cross_validation": {
      "folds": 10,
      "random_state": 42
    },
    "refresh_stability": {
      "refresh_periods": [0.3, 0.5, 0.7, 0.9]
    },
    "perturbation": {
      "perturbation_levels": [0.01, 0.05, 0.1]
    }
  }
}
```

### Environment Variables

You can also set test parameters via environment variables:

```bash
export MMM_EVAL_TRAIN_TEST_SPLIT=0.8
export MMM_EVAL_CV_FOLDS=5
export MMM_EVAL_RANDOM_STATE=42
```

## Best Practices

### Test Selection

1. **Start with accuracy**: Always run accuracy test first
2. **Add stability tests**: Include cross-validation for model comparison
3. **Consider temporal aspects**: Use refresh stability for time series
4. **Test robustness**: Include perturbation for production models

### Result Interpretation

1. **Set benchmarks**: Define acceptable performance thresholds
2. **Compare models**: Use same tests for fair comparison
3. **Consider context**: Industry-specific benchmarks may apply
4. **Monitor trends**: Track performance over time

### Continuous Testing

1. **Automate testing**: Include tests in CI/CD pipeline
2. **Regular evaluation**: Periodically reassess model performance
3. **Alert on degradation**: Set up monitoring for performance drops

## Next Steps

- Learn about [Metrics](metrics.md) to understand test outputs
- Check [Examples](examples/basic-usage.md) for practical test usage
- Review [Configuration](getting-started/configuration.md) for test customization 