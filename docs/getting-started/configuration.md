# Configuration

mmm-eval uses framework-specific configuration files to control model parameters, fitting settings, and data mappings. This guide explains how to create and use configuration files for the PyMC-Marketing framework.

## Configuration File Format

mmm-eval uses JSON configuration files. You must specify a configuration file using the `--config-path` option:

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/
```

## PyMC-Marketing Configuration Structure

A complete PyMC-Marketing configuration file has the following structure:

```json
{
  "pymc_model_config": {
    "date_column": "date_week",
    "channel_columns": ["channel_1", "channel_2"],
    "control_columns": ["price", "event_1", "event_2"],
    "adstock": "GeometricAdstock(l_max=4)",
    "saturation": "LogisticSaturation()",
    "yearly_seasonality": 2
  },
  "fit_config": {
    "target_accept": 0.9,
    "draws": 100,
    "tune": 50,
    "chains": 2,
    "random_seed": 42
  },
  "revenue_column": "revenue",
  "response_column": "quantity"
}
```

## Model Configuration (pymc_model_config)

The `pymc_model_config` section defines the PyMC model structure and parameters:

### Required Fields

#### date_column
The column name containing the date/time variable.

```json
{
  "pymc_model_config": {
    "date_column": "date_week"
  }
}
```

#### channel_columns
List of column names for marketing channels (media spend).

```json
{
  "pymc_model_config": {
    "channel_columns": ["tv_spend", "digital_spend", "print_spend"]
  }
}
```

#### adstock
The adstock transformation to apply to media channels.

```json
{
  "pymc_model_config": {
    "adstock": "GeometricAdstock(l_max=4)"
  }
}
```

Available adstock types:
- `"GeometricAdstock(l_max=4)"` - Geometric decay with maximum lag of 4
- `"WeibullAdstock(l_max=4)"` - Weibull distribution-based decay

#### saturation
The saturation transformation to apply to media channels.

```json
{
  "pymc_model_config": {
    "saturation": "LogisticSaturation()"
  }
}
```

Available saturation types:
- `"LogisticSaturation()"` - Logistic (S-curve) saturation
- `"HillSaturation()"` - Hill function saturation

### Optional Fields

#### control_columns
List of column names for control variables (optional).

```json
{
  "pymc_model_config": {
    "control_columns": ["price", "seasonality", "holiday"]
  }
}
```

#### yearly_seasonality
Number of Fourier modes for yearly seasonality (optional).

```json
{
  "pymc_model_config": {
    "yearly_seasonality": 2
  }
}
```

#### time_varying_intercept
Whether to use a time-varying intercept (default: false).

```json
{
  "pymc_model_config": {
    "time_varying_intercept": true
  }
}
```

#### time_varying_media
Whether to use time-varying media contributions (default: false).

```json
{
  "pymc_model_config": {
    "time_varying_media": true
  }
}
```

## Fit Configuration (fit_config)

The `fit_config` section defines the MCMC sampling parameters:

### Sampling Parameters

#### draws
Number of posterior samples to draw.

```json
{
  "fit_config": {
    "draws": 100
  }
}
```

#### tune
Number of tuning (warm-up) steps.

```json
{
  "fit_config": {
    "tune": 50
  }
}
```

#### chains
Number of MCMC chains to run.

```json
{
  "fit_config": {
    "chains": 2
  }
}
```

#### target_accept
Target acceptance rate for the sampler (0.0 to 1.0).

```json
{
  "fit_config": {
    "target_accept": 0.9
  }
}
```

#### random_seed
Random seed for reproducibility.

```json
{
  "fit_config": {
    "random_seed": 42
  }
}
```

### Optional Parameters

#### progress_bar
Whether to display the progress bar (default: true).

```json
{
  "fit_config": {
    "progress_bar": false
  }
}
```

#### return_inferencedata
Whether to return arviz.InferenceData (default: true).

```json
{
  "fit_config": {
    "return_inferencedata": true
  }
}
```

## Data Mapping Configuration

### revenue_column
The column name containing revenue data for ROI calculations.

```json
{
  "revenue_column": "revenue"
}
```

### response_column
The column name containing the target variable (optional, defaults to first non-date/non-channel column).

```json
{
  "response_column": "quantity"
}
```

## Example Configurations

### Minimal Configuration

```json
{
  "pymc_model_config": {
    "date_column": "date",
    "channel_columns": ["tv_spend", "digital_spend"],
    "adstock": "GeometricAdstock(l_max=4)",
    "saturation": "LogisticSaturation()"
  },
  "fit_config": {
    "target_accept": 0.9,
    "draws": 100,
    "tune": 50,
    "chains": 2,
    "random_seed": 42
  },
  "revenue_column": "revenue"
}
```

### Advanced Configuration

```json
{
  "pymc_model_config": {
    "date_column": "date_week",
    "channel_columns": ["tv_spend", "digital_spend", "print_spend", "radio_spend"],
    "control_columns": ["price", "seasonality", "holiday", "competitor_promo"],
    "adstock": "WeibullAdstock(l_max=6)",
    "saturation": "HillSaturation()",
    "yearly_seasonality": 4,
    "time_varying_intercept": true,
    "time_varying_media": false
  },
  "fit_config": {
    "target_accept": 0.95,
    "draws": 2000,
    "tune": 1000,
    "chains": 4,
    "random_seed": 123,
    "progress_bar": true,
    "return_inferencedata": true
  },
  "revenue_column": "revenue",
  "response_column": "sales"
}
```

## Configuration Best Practices

### Model Complexity

* **Start simple**: Begin with basic adstock and saturation functions
* **Add complexity gradually**: Increase seasonality terms and time-varying components as needed
* **Monitor convergence**: Use more chains and draws for complex models

### Sampling Parameters

* **More draws**: Use 1000+ draws for production models
* **Multiple chains**: Use 2-4 chains for reliable convergence
* **Adequate tuning**: Set tune to 50-100% of draws for complex models
* **Acceptance rate**: Target 0.9-0.95 for optimal sampling efficiency

### Performance Considerations

* **Data size**: Larger datasets require more sampling iterations
* **Model complexity**: More parameters increase computation time
* **Hardware**: More CPU cores can speed up multi-chain sampling

## Configuration Validation

mmm-eval validates your configuration file and will raise errors for:

* Missing required fields
* Invalid field types
* Unsupported adstock or saturation functions
* Invalid parameter ranges

## Creating Configurations Programmatically

You can also create configurations programmatically using the PyMC-Marketing library:

```python
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from mmm_eval.configs import PyMCConfig

# Create a PyMC model
model = MMM(
    date_column="date_week",
    channel_columns=["channel_1", "channel_2"],
    adstock=GeometricAdstock(l_max=4),
    saturation=LogisticSaturation(),
    yearly_seasonality=2
)

# Create configuration
config = PyMCConfig.from_model_object(
    model_object=model,
    fit_kwargs={"target_accept": 0.9, "draws": 100, "chains": 2},
    revenue_column="revenue",
    response_column="quantity"
)

# Save to JSON
config.save_model_object_to_json("./", "my_config")
```

## Next Steps

* Learn about [Data Formats](../user-guide/data-formats.md) for different data structures
* Explore [Examples](../examples/basic-usage.md) for configuration use cases
* Check the [CLI Reference](../user-guide/cli.md) for command-line options 