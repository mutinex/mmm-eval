# Basic Usage Examples

This guide provides practical examples of how to use mmm-eval for different scenarios.

## Example 1: Basic Evaluation

The simplest way to run an evaluation:

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing
```

This assumes your data has standard column names:
- `date` - Date column
- `sales` - Target variable
- `tv_spend`, `digital_spend`, `print_spend` - Media channels

## Example 2: Custom Column Names

If your data uses different column names:

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --date-column timestamp \
  --target-column revenue \
  --media-columns television,online,radio \
  --control-columns price,holiday
```

## Example 3: Configuration File

For more complex setups, use a configuration file:

```json
{
  "data": {
    "date_column": "date",
    "target_column": "sales",
    "media_columns": ["tv_spend", "digital_spend", "print_spend"],
    "control_columns": ["price", "seasonality"],
    "date_format": "%Y-%m-%d"
  },
  "tests": {
    "accuracy": {
      "train_test_split": 0.8,
      "random_state": 42
    },
    "cross_validation": {
      "folds": 5,
      "random_state": 42
    }
  },
  "output": {
    "include_plots": true,
    "plot_format": "png"
  }
}
```

Save as `config.json` and run:

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json
```

## Example 4: Specific Tests Only

Run only certain tests:

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --test-names accuracy,cross_validation
```

## Example 5: Custom Output Directory

Save results to a specific directory:

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --output-path ./my_results/ \
  --include-plots
```

## Example 6: Reproducible Results

Set a random seed for reproducible results:

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --random-state 42 \
  --train-test-split 0.8
```

## Example 7: Different Output Formats

Save results in CSV format:

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --output-format csv
```

## Example 8: Verbose Output

Get detailed information during execution:

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --verbose
```

## Example 9: Environment Variables

Set configuration using environment variables:

```bash
export MMM_EVAL_DATE_COLUMN=date
export MMM_EVAL_TARGET_COLUMN=sales
export MMM_EVAL_MEDIA_COLUMNS=tv_spend,digital_spend,print_spend
export MMM_EVAL_TRAIN_TEST_SPLIT=0.8
export MMM_EVAL_RANDOM_STATE=42

mmm-eval --input-data-path data.csv --framework pymc-marketing
```

## Example 10: Complete Workflow

A complete example with all options:

```bash
mmm-eval \
  --input-data-path marketing_data.csv \
  --framework pymc-marketing \
  --config-path evaluation_config.json \
  --date-column date \
  --target-column sales \
  --media-columns tv_spend,digital_spend,print_spend,radio_spend \
  --control-columns price,seasonality,holiday \
  --test-names accuracy,cross_validation,refresh_stability,perturbation \
  --train-test-split 0.8 \
  --cv-folds 5 \
  --random-state 42 \
  --output-path ./evaluation_results/ \
  --output-format json \
  --include-plots \
  --plot-format png \
  --verbose
```

## Data Format Examples

### Basic CSV Structure

```csv
date,sales,tv_spend,digital_spend,print_spend,price
2023-01-01,1000,5000,2000,1000,10.99
2023-01-02,1200,5500,2200,1100,10.99
2023-01-03,1100,5200,2100,1050,11.99
...
```

### With Control Variables

```csv
date,sales,tv_spend,digital_spend,print_spend,price,seasonality,holiday
2023-01-01,1000,5000,2000,1000,10.99,0.8,0
2023-01-02,1200,5500,2200,1100,10.99,0.9,0
2023-01-03,1100,5200,2100,1050,11.99,0.7,1
...
```

## Expected Output

After running an evaluation, you'll find:

```
evaluation_results/
├── results.json              # Detailed results
├── results_summary.csv       # Summary metrics
└── plots/
    ├── accuracy_plot.png     # Accuracy test plots
    ├── cross_validation_plot.png
    ├── refresh_stability_plot.png
    └── perturbation_plot.png
```

## Next Steps

- Learn about [Data Formats](user-guide/data-formats.md) for different data structures
- Explore [Advanced Scenarios](advanced-scenarios.md) for complex use cases
- Check the [Configuration](getting-started/configuration.md) guide for detailed settings 