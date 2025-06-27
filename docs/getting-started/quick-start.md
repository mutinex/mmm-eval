# Quick Start

This guide will help you get started with mmm-eval quickly by walking through an example notebook. You'll learn the evaluation workflow and how to interpret the results. To see how to run mmm-eval from the command line, check out [CLI](../user-guide/cli.md).

## Prerequisites

Before you begin, make sure you have:

1. **mmm-eval installed** - See [Installation](installation.md) if you haven't installed it yet
2. **Your MMM data** - A CSV or Parquet file with your marketing mix model data
3. **A supported framework** - Currently PyMC-Marketing is supported

## Basic Usage

### Prepare Your Data

Your data should contain the following columns (see [Data Requirements](../user-guide/data.md#data-requirements) for more)

* Date/time period
* Revenue variable (for calculating ROI)
* Marketing spend $ by channel (e.g., TV, digital, print)
* `OPTIONAL` Response variable (e.g., sales, conversions, units)
* `OPTIONAL` Control variables (e.g., price, seasonality)

Example data structure:

```csv
date_week,quantity,revenue,TV,radio,price,event_1,event_2
2023-01-01,1000,7000,5000,2000,10.99,0,0
2023-01-08,1200,8000,5500,2200,10.99,0,0
2023-01-15,1100,7500,5200,2100,11.99,1,0
2023-01-22,1300,9000,6000,2400,11.99,0,1
2023-01-29,1400,9500,6500,2600,12.99,0,0
```

### Evaluating your PyMC MMM (follow along in `mmm-eval/examples/pymc_eval.ipynb`)
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


If we look at the evaluation output ```display(results)```, we see the following table:

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

## Next Steps

Now that you've run your first evaluation:

1. **Explore the [User Guide](../user-guide/cli.md)** for detailed CLI options
2. **Check out [Examples](../examples/basic-usage.md)** for more complex scenarios
3. **Learn about [Data](../user-guide/data.md)** for different data structures
4. **Review [Configuration](../user-guide/configuration.md)** for advanced settings

## Getting Help

If you encounter issues:

* Check the [CLI Reference](../user-guide/cli.md) for all available options
* Look at [Examples](../examples/basic-usage.md) for similar use cases
* Join our [Discussions](https://github.com/Mutiny-Group/mmm-eval/discussions) for community support