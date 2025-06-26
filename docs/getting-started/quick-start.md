# Quick Start

This guide will help you get started with mmm-eval quickly. You'll learn how to run your first evaluation and understand the basic workflow.

## Prerequisites

Before you begin, make sure you have:

1. **mmm-eval installed** - See [Installation](installation.md) if you haven't installed it yet
2. **Your MMM data** - A CSV file with your marketing mix model data
3. **A configuration file** - JSON configuration for the PyMC-Marketing framework
4. **A supported framework** - Currently PyMC-Marketing is supported

## Basic Usage

### 1. Prepare Your Data

Your data should be in CSV format with columns for:

* Date/time period
* Target variable (e.g., sales, conversions)
* Revenue variable (for calculating ROI)
* Marketing spend channels (e.g., TV, digital, print)
* Control variables (e.g., price, seasonality)

Example data structure:

```csv
date_week,quantity,revenue,channel_1,channel_2,price,event_1,event_2
2023-01-01,1000,7000,5000,2000,10.99,0,0
2023-01-08,1200,8000,5500,2200,10.99,0,0
2023-01-15,1100,7500,5200,2100,11.99,1,0
2023-01-22,1300,9000,6000,2400,11.99,0,1
2023-01-29,1400,9500,6500,2600,12.99,0,0
```

### 2. Create a Configuration File

Create a JSON configuration file for the PyMC-Marketing framework:

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

Save this as `config.json`.

### 3. Run Your First Evaluation

The basic command to run an evaluation:

```bash
mmm-eval \
  --input-data-path your_data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --output-path ./results/
```

This will:

* Load your data and configuration
* Run the PyMC-Marketing framework
* Execute all available validation tests
* Save results to the specified output directory

### 4. View Results

After the evaluation completes, you'll find:

* `mmm_eval_pymc-marketing_YYYYMMDD_HHMMSS.csv` - A table of test results with metrics for each test

## Available Tests

mmm-eval runs four standard validation tests:

* **accuracy** - Model accuracy using holdout validation
* **cross_validation** - Time series cross-validation performance
* **refresh_stability** - Model stability over different time periods
* **perturbation** - Sensitivity to data perturbations

### Run Specific Tests

If you only want to run certain tests:

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --output-path ./results/ \
  --test-names accuracy cross_validation
```

### Verbose Output

Get detailed information during execution:

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --output-path ./results/ \
  --verbose
```

## Understanding Results

### Key Metrics

mmm-eval provides several key metrics:

* **MAPE** (Mean Absolute Percentage Error) - Accuracy measure
* **RMSE** (Root Mean Square Error) - Error magnitude
* **R-squared** - Model fit quality

### Test Results

Each test provides specific insights:

* **Accuracy Test**: How well the model predicts on unseen data
* **Cross-Validation**: Model performance across different time periods
* **Refresh Stability**: How consistent the model is over time
* **Perturbation**: How sensitive the model is to data changes

## Common Use Cases

### Custom Output Directory

Specify where to save results:

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --output-path ./my_results/
```

### Complete Example

A complete example with all options:

```bash
mmm-eval \
  --input-data-path marketing_data.csv \
  --framework pymc-marketing \
  --config-path evaluation_config.json \
  --test-names accuracy cross_validation refresh_stability perturbation \
  --output-path ./evaluation_results/ \
  --verbose
```

## Data Requirements

### Minimum Data Requirements

* **Time period**: At least 52 weeks (1 year) of data
* **Frequency**: Weekly or daily data (consistent frequency)
* **Observations**: Minimum 100 data points recommended
* **Media channels**: At least 2-3 channels for meaningful analysis
* **Revenue data**: Required for ROI calculations

### Data Quality Requirements

* No missing values in required columns
* Complete time series (no gaps in dates)
* Consistent date format
* Non-negative values for spend columns

## Next Steps

Now that you've run your first evaluation:

1. **Explore the [User Guide](../user-guide/cli.md)** for detailed CLI options
2. **Check out [Examples](../examples/basic-usage.md)** for more complex scenarios
3. **Learn about [Data Formats](../user-guide/data-formats.md)** for different data structures
4. **Review [Configuration](../configuration.md)** for advanced settings

## Getting Help

If you encounter issues:

* Check the [CLI Reference](../user-guide/cli.md) for all available options
* Look at [Examples](../examples/basic-usage.md) for similar use cases
* Join our [Discussions](https://github.com/Mutiny-Group/mmm-eval/discussions) for community support