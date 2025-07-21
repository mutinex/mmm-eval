# Tests

mmm-eval provides a wide-ranging suite of validation tests to evaluate MMM performance. This guide explains each test and how to interpret the results.

## Overview

mmm-eval includes six main types of validation tests:

1. **Accuracy Tests**: Measure how well the model fits the data
2. **Cross-Validation Accuracy Test**: Assess model generalization
3. **Refresh Stability Tests**: Evaluate model stability over time
4. **Robustness Tests**: Evaluate model sensitivity to data changes

## Accuracy Tests

Accuracy tests evaluate how well the model fits the data using different validation approaches.

Accuracy can be considered a necessary, but not sufficient indicator of a good model - a model
can perform well on accuracy tests but still get the causal relationships in the data wrong. However,
it is very effective for identifying poor models, as poor in-sample and/or out-of-sample performance
almost always implies that the model is failing to capture the causual structure of the problem at
hand.

### Holdout Accuracy Test

The holdout accuracy test evaluates model performance by splitting data into train/test sets and calculating metrics on the test set.

#### Process

1. **Data Split**: Data is split into training and test sets
2. **Model Training**: Model is fitted on training data
3. **Out-of-Sample Prediction**: Predictions made on held-out test data
4. **Performance Metrics**: Calculated on out-of-sample predictions

#### Metrics

- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **SMAPE (Symmetric Mean Absolute Percentage Error)**: Symmetric version of MAPE
- **R-squared**: Proportion of variance explained by the model

#### Interpretation

- **Lower MAPE**: Better model performance
- **Lower SMAPE**: Better symmetric model performance
- **Higher R-squared**: Better model fit (0-1 scale)

### In-Sample Accuracy Test

The in-sample accuracy test evaluates model performance by fitting the model on the full dataset and calculating metrics on the training data.

#### Process

1. **Full Dataset Training**: Model is fitted on the complete dataset
2. **In-Sample Prediction**: Predictions made on the same data used for training
3. **Performance Metrics**: Calculated on in-sample predictions

#### Metrics

- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **SMAPE (Symmetric Mean Absolute Percentage Error)**: Symmetric version of MAPE
- **R-squared**: Proportion of variance explained by the model

#### Interpretation

- **Lower MAPE**: Better model fit to training data
- **Lower SMAPE**: Better symmetric model fit
- **Higher R-squared**: Better explanatory power
- **Comparison with holdout**: Helps identify overfitting (much better in-sample than holdout performance)

## Cross-Validated Holdout Accuracy Test

A cross-validated version of the holdout accuracy test. The generalization performance of the
model is tested more rigorously by splitting the data into multiple train/test "folds" and
averaging over the results.

We use the leave-future-out (LFO) cross validation strategy, which is widely used for
out-of-sample testing of timeseries models. For a dataset with time indices `0, ..., T`,
this involves fitting on `[0, ..., T-X]` and testing on 
`[T-X+1, T-X+1+k]`, then incrementing `X` in order to increase the size of the training set
while keeping the test set size `k` fixed. (N.B. `X` and `k` must be strictly positive 
integers)

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

The refresh stability test evaluates how much media ROI estimates change as more data is
added to the model.

NOTE: we define ROI as `100 * (R/S - 1)`, where `R` is estimated revenue and `S` is paid
media spend for a particular media channel. Under this convntion, a ROI of 0% implies $1
spend yields a $1 return, a ROI of 100% implies $1 spend yields a $2 return, and so on.

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

## Robustness Tests

The robustness test evaluates how sensitive the model is to changes in the input data.

### Perturbation Test

The perturbation test evaluates how sensitive the model is to noise in the input data by adding Gaussian noise to media spend data and measuring the change in ROI estimates.

#### Process

1. **Baseline Model**: Train on original data
2. **Noise Addition**: Add Gaussian noise to primary regressor columns (usually spend or impressions, depending on the model spec)
3. **Retrain Model**: Fit model on noisy data
4. **Compare estimated impacts**: Compare ROI estimates across the two models
5. **Sensitivity Metrics**: Calculate percentage changes

#### Metrics

- **Percentage Change**: Change in ROI estimates for each channel
- **Channel Sensitivity**: Which channels are most sensitive to noise

#### Interpretation

- **Low percentage changes**: Robust model (good)
- **High percentage changes**: Sensitive model (may need more data or regularization)
- **Channel-specific sensitivity**: Some channels more stable than others

### Placebo Test

The placebo test (also known as a falsifiability test) evaluates whether the model can detect spurious correlations by introducing a randomly shuffled media channel and checking if the model assigns a low ROI to this spurious feature.

#### Process

1. **Channel Selection**: Randomly select an existing media channel
2. **Data Shuffling**: Randomly permute the rows of the selected channel's data to break time correlation with the target variable
3. **Model Training**: Fit the model with the shuffled channel added
4. **ROI Assessment**: Record the estimated ROI for the shuffled channel
5. **Validation**: Check if the shuffled channel ROI is appropriately low

#### Metrics

- **Shuffled Channel ROI**: Estimated ROI for the spurious channel

#### Interpretation

- **Low ROI (≤ -50%)**: Model correctly identifies spurious correlation (good)
- **High ROI (> -50%)**: Model may be overfitting or detecting spurious patterns (concerning)
- **Test Skipped**: Indicates reach and frequency regressor type not supported

#### Purpose

This test helps validate that the model is not simply memorizing patterns in the data or detecting spurious correlations. A well-performing model should assign a low ROI to a channel that has no meaningful relationship with the target variable.

## Running Tests

### All Tests (Default)

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/
```

### Specific Tests

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/ --test-names holdout_accuracy in_sample_accuracy cross_validation
```

### Available Test Names

- `holdout_accuracy`: Holdout accuracy tests only
- `in_sample_accuracy`: In-sample accuracy tests only
- `cross_validation`: Cross-validation tests only
- `refresh_stability`: Refresh stability tests only
- `perturbation`: Perturbation tests only
- `placebo`: Placebo tests only

## Test Configuration

If you'd like to modify the test pass/fail thresholds, you can fork the branch and
modify the thresholds in `mmm_eval/metrics/threshold_constants.py`.

## Interpreting Results

### Good Model Indicators

- **Holdout Accuracy**: MAPE < 15%, SMAPE < 15%, R-squared > 0.8
- **In-Sample Accuracy**: MAPE < 10%, SMAPE < 10%, R-squared > 0.9
- **Cross-Validation**: Out-of-sample MAPE/SMAPE similar to in-sample
- **Refresh Stability**: Parameter changes < 10%
- **Perturbation**: ROI changes < 5%
- **Placebo**: Shuffled channel ROI ≤ -50%

### Warning Signs

- **Poor Performance**: High MAPE/SMAPE or low R-squared
- **Overfitting**: Much better in-sample than holdout performance
- **Unstable Model**: Large parameter changes
- **Data Issues**: Missing values or extreme outliers

## Best Practices

### Test Selection

- **Start with holdout accuracy**: Always run holdout accuracy tests first
- **Add in-sample accuracy**: To assess model fit and identify overfitting
- **Include cross-validation**: For generalization assessment
- **Add stability tests**: For production models
- **Include robustness tests**: To evaluate model sensitivity to data changes

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