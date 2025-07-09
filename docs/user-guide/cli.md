# Command Line Interface

BenjaMMMin provides a command-line interface (CLI) for running MMM evaluations. This guide covers all available options and usage patterns.

## Basic Usage

The basic command structure is:

```bash
benjammmin [OPTIONS] --input-data-path PATH --framework FRAMEWORK --config-path PATH --output-path PATH
```

## Required Arguments

### --input-data-path

Path to your input data file (CSV or Parquet format).

### --framework

The MMM framework to use for evaluation.

### --config-path

Path to a framework-specific JSON configuration file.

### --output-path

Directory to save evaluation results.

## Optional Arguments

#### --test-names

Specify which validation tests to run. Can specify multiple tests by repeating the flag.

```bash
# Run all tests (default)
benjammmin --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/

# Run two tests
benjammmin --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/ --test-names accuracy cross_validation

# Available tests: accuracy, cross_validation, refresh_stability, perturbation
```

#### --verbose

Enable verbose logging for detailed output and error info.

## Complete Example

Here's a complete example with all options:

```bash
benjammmin \
  --input-data-path marketing_data.csv \
  --framework pymc-marketing \
  --config-path evaluation_config.json \
  --test-names accuracy cross_validation refresh_stability perturbation \
  --output-path ./evaluation_results/ \
  --verbose
```

## Available Tests
See [Tests](./tests.md)

## Framework Support
See [Frameworks](./frameworks.md)

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

## Help and Information

### --help

Display all available options and their descriptions:

```bash
benjammmin --help
```

### Getting Help

If you encounter issues:

- Look at at [Troubleshooting](./troubleshooting.md) for common problems and their solutions
- Check the [Configuration Guide](../getting-started/configuration.md) for config file format
- Review the [Data Guide](data.md) for data requirements
- See [Examples](../examples/basic-usage.md) for similar use cases
- Join our [Discussions](https://github.com/Mutiny-Group/mmm-eval/discussions) for community support

## Next Steps

- Learn about [Data](data.md) for different data structures
- Explore [Examples](../examples/basic-usage.md) for practical use cases
- Check the [Configuration Guide](../getting-started/configuration.md) for advanced settings 