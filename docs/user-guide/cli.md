# Command Line Interface

mmm-eval provides a command-line interface for running MMM evaluations. This guide covers all available options and usage patterns.

## Basic Usage

The basic command structure is:

```bash
mmm-eval [OPTIONS] --input-data-path PATH --framework FRAMEWORK
```

## Required Arguments

### --input-data-path

Path to your input data file (CSV format).

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing
```

### --framework

The MMM framework to use for evaluation.

```bash
# Currently supported frameworks
mmm-eval --input-data-path data.csv --framework pymc-marketing
```

## Optional Arguments

### Data Configuration

#### --config-path

Path to a JSON configuration file.

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json
```

#### --date-column

Name of the date column in your data.

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --date-column date
```

#### --target-column

Name of the target variable column.

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --target-column sales
```

#### --media-columns

Comma-separated list of media channel columns.

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --media-columns tv_spend,digital_spend,print_spend
```

#### --control-columns

Comma-separated list of control variable columns.

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --control-columns price,seasonality
```

### Test Configuration

#### --test-names

Comma-separated list of tests to run.

```bash
# Run specific tests
mmm-eval --input-data-path data.csv --framework pymc-marketing --test-names accuracy,cross_validation

# Available tests: accuracy, cross_validation, refresh_stability, perturbation
```

#### --train-test-split

Proportion of data to use for training (0.0 to 1.0).

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --train-test-split 0.8
```

#### --cv-folds

Number of cross-validation folds.

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --cv-folds 5
```

#### --random-state

Random seed for reproducibility.

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --random-state 42
```

### Output Configuration

#### --output-path

Directory to save results.

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --output-path ./results/
```

#### --output-format

Output format for results (json or csv).

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --output-format json
```

#### --include-plots

Whether to generate plots.

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --include-plots
```

#### --plot-format

Format for generated plots (png, pdf, svg).

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --plot-format png
```

### Framework-Specific Options

#### PyMC-Marketing Options

```bash
# Specify seasonality parameters
mmm-eval --input-data-path data.csv --framework pymc-marketing --yearly-seasonality 10 --weekly-seasonality 3
```

## Complete Example

Here's a complete example with all options:

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --date-column date \
  --target-column sales \
  --media-columns tv_spend,digital_spend,print_spend \
  --control-columns price,seasonality \
  --test-names accuracy,cross_validation,refresh_stability \
  --train-test-split 0.8 \
  --cv-folds 5 \
  --random-state 42 \
  --output-path ./results/ \
  --output-format json \
  --include-plots \
  --plot-format png
```

## Environment Variables

You can also set configuration using environment variables:

```bash
export MMM_EVAL_DATE_COLUMN=date
export MMM_EVAL_TARGET_COLUMN=sales
export MMM_EVAL_MEDIA_COLUMNS=tv_spend,digital_spend,print_spend
export MMM_EVAL_TRAIN_TEST_SPLIT=0.8
export MMM_EVAL_RANDOM_STATE=42

mmm-eval --input-data-path data.csv --framework pymc-marketing
```

## Help and Information

### --help

Display help information:

```bash
mmm-eval --help
```

### --version

Display version information:

```bash
mmm-eval --version
```

### --verbose

Enable verbose output:

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --verbose
```

## Error Handling

### Common Errors

1. **File not found**: Ensure the input data file exists and the path is correct
2. **Invalid column names**: Check that column names match your data file
3. **Insufficient data**: Ensure you have enough data for the specified train-test split
4. **Framework errors**: Check that all required dependencies are installed

### Debug Mode

Enable debug mode for detailed error information:

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --debug
```

## Output Files

After running an evaluation, you'll find the following files in your output directory:

- `results.json` - Detailed test results in JSON format
- `results_summary.csv` - Summary metrics in CSV format
- `plots/` - Directory containing generated plots
  - `accuracy_plot.png` - Accuracy test visualizations
  - `cross_validation_plot.png` - Cross-validation results
  - `refresh_stability_plot.png` - Refresh stability analysis
  - `perturbation_plot.png` - Perturbation test results

## Next Steps

- Learn about [Data Formats](data-formats.md) for different data structures
- Explore [Examples](examples/basic-usage.md) for practical use cases
- Check the [Configuration](getting-started/configuration.md) guide for advanced settings 