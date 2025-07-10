# Configuration

mmm-eval uses framework-specific configurations to control model parameters, fitting settings, and data mappings. This guide explains how to create and use configurations for the PyMC-Marketing framework.

## Creating Configurations

There are two ways to create a configuration (config):

1. From a model object (preferred).

!!! info
    See the Meridian notebook in the `examples/` directory for a walkthrough of how to do
    this with Meridian.

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

!!! warning "Meridian Config Creation"
    Option 2 is not supported for Meridian models due to the complexity of the model objects.

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

If you want to run mmm-eval from the CLI, you will need to save the config saved to a JSON file. We recommend Option 1 above to avoid any errors related to improper stringifiation of the model object in the manual approach.


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

### Data Mapping Configuration

### revenue_column
The column name containing revenue data for ROI calculations.

```json
{
  "revenue_column": "revenue"
}
```

### response_column
The column name containing the target variable (optional, defaults to revenue if not provided).

```json
{
  "response_column": "quantity"
}
```

## Meridian Configuration Structure

A Meridian configuration file has the following structure:

```json
{
  "input_data_builder_config": {
    "date_column": "date_week",
    "media_channels": ["tv", "digital", "print"],
    "channel_spend_columns": ["tv_spend", "digital_spend", "print_spend"],
    "response_column": "sales"
  },
  "model_spec_config": {
    "prior": "PriorDistribution(...)",
    "media_effects_dist": "log_normal",
    "max_lag": 8
  },
  "sample_posterior_config": {
    "n_chains": 4,
    "n_keep": 1000
  },
  "revenue_column": "revenue"
}
```

## Input Data Builder Configuration (input_data_builder_config)

The `input_data_builder_config` section defines how to construct the data object for the Meridian model:

### Required Fields

#### date_column
The column name containing the date/time variable.

```json
{
  "input_data_builder_config": {
    "date_column": "date_week"
  }
}
```

#### media_channels
List of media channel names (not column names).

```json
{
  "input_data_builder_config": {
    "media_channels": ["tv", "digital", "print"]
  }
}
```

#### channel_spend_columns
List of column names for media channel spend variables.

```json
{
  "input_data_builder_config": {
    "channel_spend_columns": ["tv_spend", "digital_spend", "print_spend"]
  }
}
```

#### response_column
The column name containing the target variable.

```json
{
  "input_data_builder_config": {
    "response_column": "sales"
  }
}
```

### Optional Fields

#### channel_impressions_columns
List of column names for media channel impressions variables.

```json
{
  "input_data_builder_config": {
    "channel_impressions_columns": ["tv_impressions", "digital_impressions", "print_impressions"]
  }
}
```

!!! note "Mutual Exclusion"
    `channel_impressions_columns` and `channel_reach_columns` cannot both be provided.

#### channel_reach_columns
List of column names for media channel reach variables.

```json
{
  "input_data_builder_config": {
    "channel_reach_columns": ["tv_reach", "digital_reach", "print_reach"]
  }
}
```

!!! note "Paired Fields"
    `channel_reach_columns` and `channel_frequency_columns` must be provided together (both or neither).

#### channel_frequency_columns
List of column names for media channel frequency variables.

```json
{
  "input_data_builder_config": {
    "channel_frequency_columns": ["tv_frequency", "digital_frequency", "print_frequency"]
  }
}
```

#### organic_media_columns
List of column names for organic media variables.

```json
{
  "input_data_builder_config": {
    "organic_media_columns": ["organic_search", "direct_traffic"]
  }
}
```

!!! note "Paired Fields"
    `organic_media_columns` and `organic_media_channels` must be provided together (both or neither).

#### organic_media_channels
List of channel names for organic media variables.

```json
{
  "input_data_builder_config": {
    "organic_media_channels": ["organic_search", "direct_traffic"]
  }
}
```

#### non_media_treatment_columns
List of column names for non-media treatment variables.

```json
{
  "input_data_builder_config": {
    "non_media_treatment_columns": ["price_promotion", "seasonal_event"]
  }
}
```

#### control_columns
List of column names for control variables.

```json
{
  "input_data_builder_config": {
    "control_columns": ["price", "competitor_activity", "economic_indicator"]
  }
}
```

## Model Specification Configuration (model_spec_config)

The `model_spec_config` section defines the Meridian model structure and parameters:

### Required Fields

#### prior
The prior distribution configuration. This is a complex object that defines the prior distributions for model parameters.

```json
{
  "model_spec_config": {
    "prior": "PriorDistribution(roi_m=LogNormal(0.2, 0.9), ...)"
  }
}
```

### Optional Fields

The set of optional fields matches the optional inputs to the Meridian [ModelSpec](https://developers.google.com/meridian/reference/api/meridian/model/spec/ModelSpec?hl=en) object.

## Sample Posterior Configuration (sample_posterior_config)

The `sample_posterior_config` section defines the MCMC sampling parameters for Meridian models:

### Sampling Parameters

#### n_chains
Number of MCMC chains to run (default: 4).

```json
{
  "sample_posterior_config": {
    "n_chains": 4
  }
}
```

#### n_adapt
Number of adaptation steps (default: 500).

```json
{
  "sample_posterior_config": {
    "n_adapt": 500
  }
}
```

#### n_burnin
Number of burn-in steps (default: 500).

```json
{
  "sample_posterior_config": {
    "n_burnin": 500
  }
}
```

#### n_keep
Number of posterior samples to keep (default: 1000).

```json
{
  "sample_posterior_config": {
    "n_keep": 1000
  }
}
```

Other optional arguments match the API of the [Meridian.sample_posterior function](https://developers.google.com/meridian/reference/api/meridian/model/model/Meridian?hl=en#sample_posterior.)


## Example Configurations (PyMC-marketing)

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

## Example Configurations (Meridian)

!!! warning
    As stated above, we strongly suggest you create the Meridian config objects
    programatically to ensure the config can be de-seralized properly.

### Minimal Configuration

```json
{
  "input_data_builder_config": {
    "date_column": "date_week",
    "media_channels": ["tv", "digital"],
    "channel_spend_columns": ["tv_spend", "digital_spend"],
    "response_column": "sales"
  },
  "model_spec_config": {
    "prior": "PriorDistribution(roi_m=LogNormal(0.2, 0.9))",
    "media_effects_dist": "log_normal",
    "max_lag": 8
  },
  "sample_posterior_config": {
    "n_chains": 4,
    "n_keep": 1000,
    "n_adapt": 500,
    "n_burnin": 500
  },
  "revenue_column": "revenue"
}
```

### Advanced Configuration

```json
{
  "input_data_builder_config": {
    "date_column": "date_week",
    "media_channels": ["tv", "digital", "print", "radio"],
    "channel_spend_columns": ["tv_spend", "digital_spend", "print_spend", "radio_spend"],
    "channel_reach_columns": ["tv_reach", "digital_reach", "print_reach", "radio_reach"],
    "channel_frequency_columns": ["tv_frequency", "digital_frequency", "print_frequency", "radio_frequency"],
    "organic_media_columns": ["organic_search", "direct_traffic"],
    "organic_media_channels": ["organic_search", "direct_traffic"],
    "non_media_treatment_columns": ["price_promotion", "seasonal_event"],
    "control_columns": ["price", "competitor_activity", "economic_indicator"],
    "response_column": "sales"
  },
  "model_spec_config": {
    "prior": "PriorDistribution(roi_m=LogNormal(0.2, 0.9), roi_rf=LogNormal(0.1, 0.5))",
    "media_effects_dist": "log_normal",
    "hill_before_adstock": true,
    "max_lag": 12,
    "unique_sigma_for_each_geo": true,
    "media_prior_type": "roi",
    "rf_prior_type": "roi",
    "paid_media_prior_type": "roi",
    "roi_calibration_period": [0.8, 1.2],
    "rf_roi_calibration_period": [0.9, 1.1],
    "organic_media_prior_type": "contribution",
    "organic_rf_prior_type": "contribution",
    "non_media_treatments_prior_type": "contribution",
    "non_media_baseline_values": [1.0, 0.5],
    "knots": 10,
    "baseline_geo": "US",
    "holdout_id": [1, 2, 3],
    "control_population_scaling_id": [1, 2],
    "non_media_population_scaling_id": [1, 2]
  },
  "sample_posterior_config": {
    "n_chains": 4,
    "n_adapt": 1000,
    "n_burnin": 1000,
    "n_keep": 2000,
    "max_tree_depth": 12,
    "max_energy_diff": 1000.0,
    "unrolled_leapfrog_steps": 2,
    "parallel_iterations": 20,
    "seed": 123
  },
  "revenue_column": "revenue"
}
```

## Configuration Best Practices

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

mmm-eval validates your configuration file and will raise errors for:

* Missing required fields
* Invalid field types
* Unsupported adstock or saturation functions
* Invalid parameter ranges

## Next Steps

* Learn about [Data](../user-guide/data.md) for different data structures
* Explore [Examples](../examples/basic-usage.md) for configuration use cases
* Check the [CLI Reference](../user-guide/cli.md) for command-line options 