# Quick Start

This guide will help you get started with mmm-eval quickly by walking through an example notebook. You'll learn the evaluation workflow and how to interpret the results. To see how to run mmm-eval from the command line, check out [CLI](../user-guide/cli.md).

## Prerequisites

Before you begin, make sure you have:

1. **mmm-eval installed** - See [Installation](installation.md) if you haven't installed it yet
2. **Your MMM data** - A CSV or Parquet file with your marketing mix model data
3. **A supported framework** - Currently, Meridian and PyMC-Marketing are supported

## Basic Usage

### Prepare Your Data

Your data should contain the following columns (see [Data Requirements](../user-guide/data.md#data-requirements) for more)

* Date/time period
* Revenue variable (for calculating ROI)
* Marketing spend $ by channel (e.g., TV, digital, print)
* `OPTIONAL` Response variable (e.g., sales, conversions, units). If not provided, revenue will be used as the target.
* `OPTIONAL` Control variables (e.g., price, seasonality)

!!! info "Meridian Data Inputs"
    Meridian supports multiple types of controls and media treatments, as well as a geography field. For details, see
    the Meridian notebook in the `examples/` directory for a full walkthrough and their
    documentation [here](https://developers.google.com/meridian/docs/user-guide/supported-data-types-formats?hl=en).

Example data structure:

```csv
date_week,quantity,revenue,TV,radio,price,event_1,event_2
2023-01-01,1000,7000,5000,2000,10.99,0,0
2023-01-08,1200,8000,5500,2200,10.99,0,0
2023-01-15,1100,7500,5200,2100,11.99,1,0
2023-01-22,1300,9000,6000,2400,11.99,0,1
2023-01-29,1400,9500,6500,2600,12.99,0,0
```

### Evaluating a PyMC MMM (follow along in `examples/pymc_eval.ipynb`)
First, load your data
```jupyter
data = pd.read_csv("data/example_data.csv")
```

and fit a PyMC-Marketing MMM
```jupyter
X = data.drop(columns=["revenue","quantity"])
y = data["quantity"]

model = MMM(
    date_column="date_week" ,
    channel_columns=["TV","radio"],
    adstock=GeometricAdstock(l_max=4),
    saturation=LogisticSaturation()
)

model.fit(X=X, y=y, chains=4, target_accept=0.85)
```

#### Now we evaluate! Just create a config
```jupyter
fit_kwargs = { 
    "chains": 4,
    "target_accept": 0.85,
}

config = PyMCConfig.from_model_object(base_model, fit_kwargs=fit_kwargs, response_column="quantity", revenue_column="revenue")

# Save this for later if you want to run from CLI!
config.save_model_object_to_json(save_path="data/", file_name="saved_config")
```

And we can run the evaluation suite, which returns a dataframe.
```jupyter
result = run_evaluation(framework="pymc-marketing", config=config, data=data)
```

### Evaluating a Meridian MMM (follow along in `examples/meridian_eval.ipynb`)

#### Load data and convert to Meridian data object

```jupyter
df = pd.read_excel(
    'https://github.com/google/meridian/raw/main/meridian/data/simulated_data/xlsx/geo_media.xlsx',
    engine='openpyxl',
)builder = (
    data_builder.DataFrameInputDataBuilder(kpi_type='non_revenue')
        .with_kpi(df, kpi_col="conversions")
        .with_revenue_per_kpi(df, revenue_per_kpi_col="revenue_per_conversion")
        .with_population(df)
        .with_controls(df, control_cols=["GQV", "Discount", "Competitor_Sales"])
)
channels = ["Channel0", "Channel1", "Channel2", "Channel3", "Channel4", "Channel5"]
builder = builder.with_media(
    df,
    media_cols=[f"{channel}_impression" for channel in channels],
    media_spend_cols=[f"{channel}_spend" for channel in channels],
    media_channels=channels,
)

data = builder.build()
```

#### Define a Meridian MMM
```jupyter
roi_mu = 0.2     # Mu for ROI prior for each media channel.
roi_sigma = 0.9  # Sigma for ROI prior for each media channel.
prior = prior_distribution.PriorDistribution(
    roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name=constants.ROI_M)
)
model_spec = spec.ModelSpec(prior=prior)
# sampling from the posterior is not required prior to evaluation
mmm = model.Meridian(input_data=data, model_spec=model_spec)
```

#### Set up Meridian input data builder config
```jupyter
data_preproc = df.copy()
data_preproc["revenue"] = data_preproc["revenue_per_conversion"]*data_preproc["conversions"]

channels = ["Channel0", "Channel1", "Channel2", "Channel3", "Channel4", "Channel5"]
input_data_builder_config = MeridianInputDataBuilderSchema(
    date_column="time",
    media_channels=channels,
    channel_spend_columns=[f"{col}_spend" for col in channels],
    channel_impressions_columns=[f"{col}_impression" for col in channels],
    response_column="conversions",
    control_columns=["GQV", "Competitor_Sales", "Discount"],
)
```

#### Create a config and evaluate

```
# specify a larger number of samples if you want quality results
sample_posterior_kwargs = dict(n_chains=1, n_adapt=10, n_burnin=10, n_keep=10)
config = MeridianConfig.from_model_object(mmm, input_data_builder_config=input_data_builder_config,
                                          revenue_column="revenue", sample_posterior_kwargs=sample_posterior_kwargs)
# Run the evaluation suite!
result = run_evaluation(framework="meridian", config=config, data=data_preproc)
```
#### ✨ Done ✨

## What's in `result`?

The evaluation suite runs 4 tests, each of which answers a distinct question about the quality of your model: 

* **Accuracy Test**: "How well does my model predict on unseen data?"
* **Cross-Validation**: "How *consistent* are my model's predictions across different splits of unseen data?"
* **Refresh Stability**: "How much does marketing attribution change when I add new data to my model?"
* **Perturbation**: "How sensitive is my model is to noise in the marketing inputs?"

Details on the implementation of the tests can be found in [Tests](../user-guide/tests.md). For each test, we compute multiple metrics to give as much insight into the test result as possible. These can be viewed in detail in [Metrics](../user-guide/metrics.md). For example:

* **MAPE (Mean Absolute Percentage Error)**  
  `MAPE = (100 / n) * Σ |(y_i - ŷ_i) / y_i|`

* **R-squared (Coefficient of Determination)**  
  `R² = 1 - (Σ (y_i - ŷ_i)^2) / (Σ (y_i - ȳ)^2)`


If we look at the evaluation output ```display(results)```, we'll see something like the following:

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


Notice that our model is failing every test. Seems we have some work to do!

## Changing the Thresholds
Default metric thresholds in `mmm_eval/metrics/threshold_constants.py` can be overwritten in-place to change the pass/fail cutoff for each metric.

## Troubleshooting

### Common Issues

1. **Data Format**: Ensure your data has the required columns and proper format
2. **Configuration Errors**: Check that your config file is valid JSON
3. **Memory Issues**: For large datasets, try reducing the number of chains or draws

### Getting Help

If you encounter issues:

* Check the [CLI Reference](../user-guide/cli.md) for all available options
* Look at [Examples](../examples/basic-usage.md) for similar use cases
* Join our [Discussions](https://github.com/mutinex/mmm-eval/discussions) for community support

## Next Steps

Now that you've run your first evaluation:

1. **Explore the [User Guide](../user-guide/cli.md)** for detailed CLI options
2. **Check out [Examples](../examples/basic-usage.md)** for more complex scenarios
3. **Learn about [Data](../user-guide/data.md)** for different data structures
4. **Review [Configuration](../getting-started/configuration.md)** for advanced settings