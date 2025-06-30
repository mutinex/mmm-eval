# Google Meridian Integration

mmm-eval provides comprehensive support for Google's Meridian MMM framework, allowing you to evaluate Meridian models using the same standardized validation tests and metrics as other supported frameworks.

## Overview

Google Meridian is an open-source Marketing Mix Modeling (MMM) tool built on TensorFlow Probability. It provides a Bayesian framework for modeling marketing effectiveness with features like:

- **Geo-hierarchical modeling** for location-specific insights
- **Time-varying parameters** for dynamic market conditions
- **Built-in ROI priors** for realistic parameter constraints
- **TensorFlow Probability backend** for efficient Bayesian inference

## Installation

To use the Meridian adapter, you'll need to install the Meridian package:

```bash
pip install meridian
```

Note: The mmm-eval framework will work without Meridian installed, but will use placeholder implementations for demonstration purposes.

## Quick Start

Here's a basic example of how to use the Meridian adapter:

```python
import pandas as pd
from mmm_eval import (
    Evaluator,
    MeridianConfig,
    MeridianPriorDistributionSchema,
    MeridianModelSpecSchema,
    MeridianModelSchema,
    MeridianFitSchema,
)
from mmm_eval.data.constants import InputDataframeConstants

# Create your data
data = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=100, freq="D"),
    "tv_spend": np.random.uniform(1000, 5000, 100),
    "digital_spend": np.random.uniform(500, 2000, 100),
    "response": np.random.normal(100, 20, 100),
    "revenue": np.random.normal(100, 20, 100),
})

# Create Meridian configuration
prior_config = MeridianPriorDistributionSchema(
    roi_mu=0.0,
    roi_sigma=1.0,
    name="roi_m"
)

model_spec_config = MeridianModelSpecSchema(
    prior=prior_config,
    media_effects_dist="log_normal",           # Distribution for media effects
    hill_before_adstock=False,                 # Apply Hill transformation before adstock
    max_lag=8,                                 # Maximum lag for adstock
    unique_sigma_for_each_geo=False,           # Unique sigma per geography
    media_prior_type="log_normal",             # Prior type for media variables
    organic_media_prior_type="contribution",   # Prior type for organic media
    non_media_treatments_prior_type="contribution",  # Prior type for non-media treatments
    knots=10,                                  # Knots for spline transformations
    baseline_geo="US",                         # Baseline geography
)

model_config = MeridianModelSchema(
    date_column="date",
    media_columns=["tv_spend", "digital_spend"],
    response_column="response",
)

fit_config = MeridianFitSchema(
    n_chains=4,          # Number of MCMC chains
    n_adapt=500,         # Number of adaptation steps
    n_burnin=500,        # Number of burn-in steps
    n_keep=1000,         # Number of posterior samples to keep
    seed=42,             # Random seed for reproducibility
)

config = MeridianConfig(
    meridian_model_config=model_config,
    model_spec_config=model_spec_config,
    fit_config=fit_config,
    revenue_column="revenue",
    response_column="response",
)

# Evaluate the model
evaluator = Evaluator(data=data)
results = evaluator.evaluate_framework("meridian", config)
```

## Configuration

The Meridian integration uses several Pydantic schemas to configure different aspects of the model:

### MeridianPriorDistributionSchema

Configures the prior distribution for ROI parameters:

```python
prior_config = MeridianPriorDistributionSchema(
    roi_mu=0.0,        # Mean of log-normal distribution
    roi_sigma=1.0,     # Standard deviation of log-normal distribution
    name="roi_m"       # Parameter name
)
```

### MeridianModelSpecSchema

Configures the model specification:

```python
model_spec_config = MeridianModelSpecSchema(
    prior=prior_config,
    media_effects_dist="log_normal",           # Distribution for media effects
    hill_before_adstock=False,                 # Apply Hill transformation before adstock
    max_lag=8,                                 # Maximum lag for adstock
    unique_sigma_for_each_geo=False,           # Unique sigma per geography
    media_prior_type="log_normal",             # Prior type for media variables
    organic_media_prior_type="contribution",   # Prior type for organic media
    non_media_treatments_prior_type="contribution",  # Prior type for non-media treatments
    knots=10,                                  # Knots for spline transformations
    baseline_geo="US",                         # Baseline geography
)
```

### MeridianModelSchema

Configures the model parameters:

```python
model_config = MeridianModelSchema(
    date_column="date",                    # Date column name
    media_columns=["tv", "digital"],       # Media channel columns
    response_column="response",            # Response variable
    control_columns=["control1", "control2"],  # Optional control variables
    geo_column="region",                   # Optional geographic column
    seasonality_columns=["season1"],       # Optional seasonality columns
)
```

### MeridianFitSchema

Configures the MCMC fitting parameters:

```python
fit_config = MeridianFitSchema(
    n_chains=4,          # Number of MCMC chains
    n_adapt=500,         # Number of adaptation steps
    n_burnin=500,        # Number of burn-in steps
    n_keep=1000,         # Number of posterior samples to keep
    seed=42,             # Random seed for reproducibility
)
```

## Advanced Usage

### Custom Prior Distributions

You can customize the prior distributions based on your domain knowledge:

```python
# Conservative ROI priors
prior_config = MeridianPriorDistributionSchema(
    roi_mu=-0.5,      # Lower mean for conservative estimates
    roi_sigma=0.5,    # Tighter standard deviation
    name="roi_m"
)
```

### Advanced ModelSpec Configuration

The ModelSpec supports many advanced parameters for fine-tuning your model:

```python
model_spec_config = MeridianModelSpecSchema(
    prior=prior_config,
    # Media effects configuration
    media_effects_dist="log_normal",           # or "normal", "gamma"
    hill_before_adstock=False,                 # Apply Hill transformation before adstock
    
    # Adstock configuration
    max_lag=8,                                 # Maximum lag for adstock transformation
    
    # Geographic modeling
    unique_sigma_for_each_geo=False,           # Use unique sigma per geography
    baseline_geo="US",                         # Baseline geography for comparisons
    
    # Prior type configurations
    media_prior_type="log_normal",             # Prior for paid media
    organic_media_prior_type="contribution",   # Prior for organic media
    non_media_treatments_prior_type="contribution",  # Prior for non-media treatments
    
    # Calibration periods
    roi_calibration_period=[0.8, 1.2],        # ROI calibration range
    rf_roi_calibration_period=[0.9, 1.1],     # RF ROI calibration range
    
    # Spline transformations
    knots=10,                                  # Number of knots for splines
    
    # Holdout and scaling
    holdout_id=[1, 2, 3],                      # Holdout period identifiers
    control_population_scaling_id=[1, 2],      # Control scaling identifiers
)
```

### Geographic Modeling

Meridian supports geographic segmentation:

```python
model_config = MeridianModelSchema(
    date_column="date",
    media_columns=["tv_spend", "digital_spend"],
    response_column="response",
    geo_column="region",  # Geographic column
)
```

### Control Variables

Include control variables to account for external factors:

```python
model_config = MeridianModelSchema(
    date_column="date",
    media_columns=["tv_spend", "digital_spend"],
    response_column="response",
    control_columns=["competitor_spend", "seasonality", "economic_index"],
)
```

## Validation Tests

The Meridian adapter supports all standard mmm-eval validation tests:

- **Accuracy Test**: Measures prediction accuracy using holdout validation
- **Cross-Validation Test**: Time-series cross-validation for robust performance assessment
- **Refresh Stability Test**: Evaluates model stability across different time periods
- **Perturbation Test**: Tests model robustness to data perturbations

## ROI Calculation

The Meridian adapter calculates channel ROIs using the same methodology as other adapters:

1. **Channel Contributions**: Extracts media channel contributions from the fitted model
2. **Revenue Conversion**: Converts response units to revenue using average revenue per unit
3. **ROI Calculation**: Computes ROI as (revenue - spend) / spend * 100%

## Comparison with Other Frameworks

Based on the [comparison analysis](https://khandelwal-shekhar.medium.com/pymc-marketing-vs-google-meridian-a-deep-dive-into-modern-marketing-mix-modeling-tools-c2c10f39200c), Meridian offers:

### Strengths
- **Google Ecosystem Integration**: Seamless integration with Google Ads and other Google products
- **Geo-hierarchical Modeling**: Built-in support for geographic segmentation
- **Operational Efficiency**: Optimized for large-scale deployments
- **Standardized API**: Opinionated approach reduces configuration complexity

### Considerations
- **Limited Customization**: Less flexible than PyMC-Marketing for custom modeling needs
- **Google Dependencies**: Tightly coupled with Google's technology stack
- **Newer Framework**: Less mature ecosystem compared to established alternatives

## Best Practices

### Data Preparation
1. **Ensure data quality**: Clean and validate your data before modeling
2. **Handle missing values**: Meridian may have specific requirements for missing data
3. **Scale control variables**: Consider scaling control variables to [-1, 1] range
4. **Check for zero-spend channels**: The adapter automatically handles channels with zero spend

### Model Configuration
1. **Start with default priors**: Use conservative default priors unless you have strong domain knowledge
2. **Validate model convergence**: Check MCMC diagnostics for proper convergence
3. **Use appropriate sample sizes**: Ensure sufficient posterior samples for reliable inference
4. **Consider geographic structure**: Leverage Meridian's geo-hierarchical capabilities when applicable

### Evaluation
1. **Run multiple tests**: Use all available validation tests for comprehensive evaluation
2. **Compare across frameworks**: Compare Meridian results with other frameworks when possible
3. **Monitor performance**: Track model performance over time to detect degradation
4. **Validate business logic**: Ensure ROI estimates align with business expectations

## Troubleshooting

### Common Issues

**Import Error**: If you see `ImportError: No module named 'meridian'`, install Meridian:
```bash
pip install meridian
```

**Placeholder Warnings**: If you see warnings about placeholder implementations, this means Meridian isn't installed or there's an issue with the import.

**Configuration Errors**: Ensure all required fields are provided in your configuration schemas.

### Getting Help

- Check the [Meridian documentation](https://developers.google.com/meridian)
- Review the [example code](../examples/meridian_example.py)
- Consult the [PyMC-Marketing vs Meridian comparison](https://www.pymc-marketing.io/en/stable/guide/mmm/comparison.html)

## Next Steps

- Explore the [full example](../examples/meridian_example.py) for detailed usage
- Learn about [validation tests](tests.md) to understand evaluation metrics
- Review [configuration options](getting-started/configuration.md) for advanced setup
- Check [best practices](best-practices.md) for optimal results 