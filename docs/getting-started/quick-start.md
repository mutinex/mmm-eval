# Quick Start

This guide will help you get started with mmm-eval quickly. You'll learn how to run your first evaluation and understand the basic workflow.

## Prerequisites

Before you begin, make sure you have:

1. **mmm-eval installed** - See [Installation](installation.md) if you haven't installed it yet
2. **Your MMM data** - A CSV file with your marketing mix model data
3. **A supported framework** - Currently PyMC-Marketing is supported

## Basic Usage

### 1. Prepare Your Data

Your data should be in CSV format with columns for:
- Date/time period
- Target variable (e.g., sales, conversions)
- Marketing channels (e.g., TV, digital, print)
- Other variables (e.g., price, seasonality)

Example data structure:
```csv
date,sales,tv_spend,digital_spend,print_spend,price
2023-01-01,1000,5000,2000,1000,10.99
2023-01-02,1200,5500,2200,1100,10.99
...
```

### 2. Run Your First Evaluation

The simplest way to run an evaluation:

```bash
mmm-eval --input-data-path your_data.csv --framework pymc-marketing
```

This will:
- Load your data
- Run the PyMC-Marketing framework
- Execute all available tests
- Save results to the current directory

### 3. View Results

After the evaluation completes, you'll find:
- `results.json` - Detailed test results
- `results_summary.csv` - Summary metrics
- `plots/` directory - Visualization plots

## Common Use Cases

### Run Specific Tests

If you only want to run certain tests:

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --test-names accuracy cross_validation
```

Available tests:
- `accuracy` - Model accuracy metrics
- `cross_validation` - Cross-validation performance
- `refresh_stability` - Model stability over time
- `perturbation` - Sensitivity to data changes

### Custom Configuration

Use a configuration file for more control:

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json
```

Example configuration:
```json
{
  "data": {
    "date_column": "date",
    "target_column": "sales",
    "media_columns": ["tv_spend", "digital_spend", "print_spend"]
  },
  "tests": {
    "accuracy": {
      "train_test_split": 0.8
    },
    "cross_validation": {
      "folds": 5
    }
  }
}
```

### Custom Output Directory

Specify where to save results:

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --output-path ./my_results/
```

## Understanding Results

### Key Metrics

mmm-eval provides several key metrics:

- **MAPE** (Mean Absolute Percentage Error) - Accuracy measure
- **RMSE** (Root Mean Square Error) - Error magnitude
- **R-squared** - Model fit quality
- **MAE** (Mean Absolute Error) - Average error

### Test Results

Each test provides specific insights:

- **Accuracy Test**: How well the model predicts on unseen data
- **Cross-Validation**: Model performance across different data splits
- **Refresh Stability**: How consistent the model is over time
- **Perturbation**: How sensitive the model is to data changes

## Next Steps

Now that you've run your first evaluation:

1. **Explore the [User Guide](user-guide/cli.md)** for detailed CLI options
2. **Check out [Examples](examples/basic-usage.md)** for more complex scenarios
3. **Learn about [Data Formats](user-guide/data-formats.md)** for different data structures
4. **Review [Metrics](user-guide/metrics.md)** to understand the results better

## Getting Help

If you encounter issues:

- Check the [CLI Reference](user-guide/cli.md) for all available options
- Look at [Examples](examples/basic-usage.md) for similar use cases
- Join our [Discussions](https://github.com/mutinex/mmm-eval/discussions) for community support 