# Configuration

BenjaMMMin uses framework-specific configurations to control model parameters, fitting settings, and data mappings. This guide explains how to create and use configurations for the PyMC-Marketing framework.

## Creating Configurations

There are two ways to create a configuration (config):

1. From a model object (preferred).

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

# Save to JSON if you want
config.save_model_object_to_json("./", "my_config")
```
2. Manually, in a JSON file.
```json
{
  "pymc_model_config": {
    "date_column": "date_week",
    "channel_columns": ["channel_1", "channel_2"],
    "adstock": "GeometricAdstock(l_max=4)",
    "saturation": "LogisticSaturation()",
  },
  "fit_config": {
    "target_accept": 0.9,
    "chains": 2,
  },
  "revenue_column": "revenue",
}
```

If you want to run BenjaMMMin from the CLI, you will need to save the config saved to a JSON file. We recommend Option 1 above to avoid any errors related to improper stringifiation of the model object in the manual approach.


## Using a Configuration

If you have the config created from the model object, you can pass that directly to the evaluation suite (see [Quick Start](../getting-started/quick-start.md)).

Alternately, if you have the config saved to a JSON, you can pass the filepath via the CLI.

```bash
benjammmin --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/
```

If you have a saved config and you're in a notebook, you can load the config from the path, then run the evaluation.
```python
new_config = PyMCConfig.load_model_config_from_json("path/to/config.json")
results = run_evaluation(new_config, ...)
```

## PyMC-Marketing Configuration Structure

A PyMC-Marketing configuration file has the following structure:

```json
{
  "pymc_model_config": {
    "date_column": "date_week",
    "channel_columns": ["channel_1", "channel_2"],
    "adstock": "GeometricAdstock(l_max=4)",
    "saturation": "LogisticSaturation()",
  },
  "fit_config": {
    // Optional 
  },
  "revenue_column": "revenue",
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
The adstock transformation to apply to media channels. Must be a valid member of the [AdstockTransformation](https://github.com/pymc-labs/pymc-marketing/blob/b17fa2bbfb55703133debe8bbd9aff00a6ffc870/pymc_marketing/mmm/components/adstock.py#L83C7-L83C28) class.

```json
{
  "pymc_model_config": {
    "adstock": "GeometricAdstock(l_max=4)"
  }
}
```

#### saturation
The saturation transformation to apply to media channels. Must be valid member of the [SaturationTransformation](https://github.com/pymc-labs/pymc-marketing/blob/b17fa2bbfb55703133debe8bbd9aff00a6ffc870/pymc_marketing/mmm/components/saturation.py#L104) class.

```json
{
  "pymc_model_config": {
    "saturation": "LogisticSaturation()"
  }
}
```

### Optional Fields
The set of optional inputs matches the optional inputs to the [MMM](https://github.com/pymc-labs/pymc-marketing/blob/b17fa2bbfb55703133debe8bbd9aff00a6ffc870/pymc_marketing/mmm/mmm.py#L67) class in PyMC.

## Fit Configuration (fit_config)

The `fit_config` section defines the MCMC sampling parameters. These parameters will be passed to `.fit()`. Note, we do not require the `X` and `y` inputs as we derive those from the data you provide. Therefore, all parameters in this config are optional.

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

#### progress_bar
Whether to display the progress bar (default: false).

```json
{
  "fit_config": {
    "progress_bar": false
  }
}
```

#### return_inferencedata
Whether to return arviz.InferenceData (default: false).

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
It is recommended to create the config programmatically and then save to JSON with our built-in methods. This ensures proper stringification of the PyMC model objects (eg. adstock and saturation functions), reducing the risk of errors when loading the config and fitting models within the evaluation suite. Example:

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

### Sampling Parameters

* **More draws**: Use 1000+ draws for production models
* **Multiple chains**: Use 2-4 chains for reliable convergence
* **Adequate tuning**: Set `tune` to 50-100% of draws for complex models
* **Acceptance rate**: Target 0.9-0.95 for optimal sampling efficiency

### Performance Considerations

* **Data size**: Larger datasets require more sampling iterations
* **Model complexity**: More parameters increase computation time
* **Hardware**: More CPU cores can speed up multi-chain sampling

## Configuration Validation

BenjaMMMin validates your configuration file and will raise errors for:

* Missing required fields
* Invalid field types
* Unsupported adstock or saturation functions
* Invalid parameter ranges

## Next Steps

* Learn about [Data](../user-guide/data.md) for different data structures
* Explore [Examples](../examples/basic-usage.md) for configuration use cases
* Check the [CLI Reference](../user-guide/cli.md) for command-line options 