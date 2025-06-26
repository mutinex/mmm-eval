# Command Line Interface

mmm-eval provides a command-line interface for running MMM evaluations. This guide covers all available options and usage patterns.

## Basic Usage

The basic command structure is:

```bash
mmm-eval [OPTIONS] --input-data-path PATH --framework FRAMEWORK --config-path PATH --output-path PATH
```

## Required Arguments

### --input-data-path

Path to your input data file (CSV or Parquet format).

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/
```

### --framework

The MMM framework to use for evaluation.

```bash
# Currently supported frameworks
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/
```

### --config-path

Path to a framework-specific JSON configuration file.

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/
```

### --output-path

Directory to save evaluation results.

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path ./results/
```

## Optional Arguments

### Test Configuration

#### --test-names

Specify which validation tests to run. Can specify multiple tests by repeating the flag.

```bash
# Run specific tests
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/ --test-names accuracy cross_validation

# Run all tests (default)
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/

# Available tests: accuracy, cross_validation, refresh_stability, perturbation
```

#### --verbose

Enable verbose logging for detailed output.

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/ --verbose
```

## Available Tests

mmm-eval provides four standard validation tests:

### accuracy
Model accuracy using holdout validation. Splits data into train/test sets and evaluates prediction performance.

### cross_validation
Time series cross-validation performance. Uses rolling window validation to assess model stability over time.

### refresh_stability
Model stability over different time periods. Tests how consistent model parameters are when refitting on different data subsets.

### perturbation
Sensitivity to data perturbations. Tests how robust the model is to small changes in the input data.

## Framework Support

### PyMC-Marketing

Currently, mmm-eval supports the PyMC-Marketing framework:

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/
```

PyMC-Marketing requires a specific configuration format. See the [Configuration Guide](../getting-started/configuration.md) for details.

## Complete Example

Here's a complete example with all options:

```bash
mmm-eval \
  --input-data-path marketing_data.csv \
  --framework pymc-marketing \
  --config-path evaluation_config.json \
  --test-names accuracy cross_validation refresh_stability perturbation \
  --output-path ./evaluation_results/ \
  --verbose
```

## Output Files

After running an evaluation, you'll find the following files in your output directory:

- `mmm_eval_{framework}_{timestamp}.csv` - Detailed test results in CSV format

### Results File Structure

The results CSV file contains the following columns:

- `test_name` - Name of the validation test
- `metric_name` - Name of the metric calculated
- `metric_value` - Value of the metric
- `metric_pass` - Whether the metric passed its threshold (if applicable)

### Example Results

```csv
test_name,metric_name,metric_value,metric_pass
accuracy,mape,0.15,True
accuracy,r_squared,0.85,True
cross_validation,mape,0.18,True
cross_validation,r_squared,0.82,True
refresh_stability,mean_percentage_change_for_each_channel:channel_1,0.05,True
refresh_stability,mean_percentage_change_for_each_channel:channel_2,0.03,True
perturbation,percentage_change_for_each_channel:channel_1,0.02,True
perturbation,percentage_change_for_each_channel:channel_2,0.01,True
```

## Error Handling

### Common Errors

1. **File not found**: Ensure the input data file and config file exist and paths are correct
2. **Invalid configuration**: Check that your JSON config file follows the required format
3. **Missing columns**: Ensure your data contains all columns specified in the configuration
4. **Framework errors**: Check that all required dependencies are installed

### Debug Mode

Enable verbose mode for detailed error information:

```bash
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/ --verbose
```

## Help and Information

### --help

Display help information:

```bash
mmm-eval --help
```

This shows all available options and their descriptions.

## Data Requirements

### Supported Formats

- **CSV**: Comma-separated values (recommended)
- **Parquet**: Apache Parquet format

### Data Structure

Your data should contain:

- **Date column**: Time series data with consistent date format
- **Target column**: The variable you want to predict (e.g., sales, conversions)
- **Revenue column**: Revenue data for calculating ROI and efficiency metrics
- **Media columns**: Marketing channel spend or activity data
- **Control columns** (optional): Additional variables that may affect the target

### Data Quality

- No missing values in required columns
- Complete time series (no gaps in dates)
- Consistent date format
- Non-negative values for spend columns

## Performance Considerations

### Computation Time

Evaluation time depends on:

- **Data size**: Larger datasets take longer to process
- **Model complexity**: More parameters increase computation time
- **Number of tests**: Running all tests takes longer than specific tests
- **Hardware**: CPU cores and memory affect speed

### Memory Usage

- **Data size**: Larger datasets require more memory
- **Model complexity**: Complex models use more memory during fitting
- **Sampling parameters**: More draws and chains increase memory usage

## Best Practices

### Configuration

- Start with simple model configurations
- Use appropriate sampling parameters for your data size
- Validate your configuration file before running evaluations

### Data Preparation

- Clean and validate your data before running evaluations
- Ensure consistent date formats and column names
- Check for missing values and outliers

### Testing Strategy

- Start with basic tests (accuracy, cross_validation)
- Add stability and perturbation tests for comprehensive evaluation
- Use verbose mode to monitor progress and identify issues

## Troubleshooting

### Common Issues

1. **Slow performance**: Reduce sampling parameters or use fewer tests
2. **Memory errors**: Reduce data size or model complexity
3. **Convergence issues**: Adjust sampling parameters or model configuration
4. **File permission errors**: Check write permissions for output directory

### Getting Help

If you encounter issues:

- Check the [Configuration Guide](../getting-started/configuration.md) for config file format
- Review the [Data Formats Guide](data-formats.md) for data requirements
- Look at [Examples](../examples/basic-usage.md) for similar use cases
- Join our [Discussions](https://github.com/Mutiny-Group/mmm-eval/discussions) for community support

## Next Steps

- Learn about [Data Formats](data-formats.md) for different data structures
- Explore [Examples](../examples/basic-usage.md) for practical use cases
- Check the [Configuration Guide](../getting-started/configuration.md) for advanced settings 