# Quick Start

This guide will help you get started with BenjaMMMin quickly by walking through an example notebook. You'll learn the evaluation workflow and how to interpret the results. To see how to run BenjaMMMin from the command line, check out [CLI](../user-guide/cli.md).

## Prerequisites

1. **BenjaMMMin installed** - See [Installation](installation.md) if you haven't installed it yet
2. **Jupyter Notebook** - For interactive examples
3. **Sample data** - We'll use synthetic data for this example

## Quick Evaluation

### Evaluating your PyMC MMM (follow along in `benjammmin/examples/pymc_eval.ipynb`)
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

If you need help:

* Check the [User Guide](../user-guide/cli.md) for detailed CLI usage
* Review the [API Reference](../api/core.md) for programmatic usage
* Join our [Discussions](https://github.com/Mutiny-Group/mmm-eval/discussions) for community support