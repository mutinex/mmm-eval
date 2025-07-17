# CLI Reference

mmm-eval provides a command-line interface (CLI) for running MMM evaluations. This guide covers all available options and usage patterns.

## Basic Usage

### Command Structure

```bash
mmm-eval [OPTIONS] --input-data-path PATH --framework FRAMEWORK --config-path PATH --output-path PATH
```

### Required Arguments

- `--input-data-path`: Path to your input data file (CSV or Parquet)
- `--framework`: MMM framework to use (`pymc-marketing` or `meridian`)
- `--config-path`: Path to your configuration file (JSON)
- `--output-path`: Directory where results will be saved

### Example Commands

```bash
# Basic evaluation
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/

# Run specific tests only
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/ --test-names holdout_accuracy in_sample_accuracy cross_validation
```

## Command Options

### Input Options

- `--input-data-path`: Path to input data file (required)
- `--config-path`: Path to configuration file (required)
- `--framework`: MMM framework to use (required)
  - Options: `pymc-marketing`, `meridian`

### Output Options

- `--output-path`: Directory for output files (required)
- `--test-names`: Specific tests to run (optional)
  - Options: `holdout_accuracy`, `in_sample_accuracy`, `cross_validation`, `refresh_stability`, `perturbation`
  - Default: All tests

### Advanced Options

- `--random-seed`: Random seed for reproducibility (optional)
- `--verbose`: Enable verbose output (optional)
- `--help`: Show help message

## Examples

### Basic Evaluation

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --output-path results/
```

### Run Specific Tests

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --output-path results/ \
  --test-names holdout_accuracy in_sample_accuracy cross_validation
```

### With Custom Random Seed

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --output-path results/ \
  --random-seed 42
```

### Verbose Output

```bash
mmm-eval \
  --input-data-path data.csv \
  --framework pymc-marketing \
  --config-path config.json \
  --output-path results/ \
  --verbose
```

## Output Structure

The CLI creates the following output structure:

```
results/
├── holdout_accuracy/
│   ├── metrics.json
│   └── plots/
├── in_sample_accuracy/
│   ├── metrics.json
│   └── plots/
├── cross_validation/
│   ├── metrics.json
│   └── plots/
├── refresh_stability/
│   ├── metrics.json
│   └── plots/
├── perturbation/
│   ├── metrics.json
│   └── plots/
└── summary.json
```

## Error Handling

### Common Errors

1. **File not found**: Ensure all file paths are correct
2. **Invalid configuration**: Check your config file format
3. **Framework not supported**: Verify framework name
4. **Data format issues**: Check data requirements

### Getting Help

```bash
# Show help
mmm-eval --help

# Show version
mmm-eval --version
```

## Next Steps

- Learn about [Data Requirements](../user-guide/data.md) for input format
- Check [Configuration](../getting-started/configuration.md) for setup
- Explore [Examples](../examples/basic-usage.md) for use cases 