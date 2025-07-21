# mmm-eval

A comprehensive evaluation framework for Marketing Mix Modeling (MMM) frameworks.

## Overview

mmm-eval provides a standardized way to evaluate and compare different MMM frameworks using a comprehensive suite of validation tests including holdout accuracy, in-sample accuracy, cross-validation, refresh stability, perturbation, and placebo metrics.

## Features

- **Multi-framework support**: Evaluate PyMC-Marketing and Google Meridian models
- **Comprehensive testing**: Holdout accuracy, in-sample accuracy, cross-validation, refresh stability, perturbation, and placebo tests
- **Standardized metrics**: Consistent evaluation across different frameworks
- **Easy to use**: Simple CLI interface and Python API
- **Extensible**: Add new frameworks and tests easily

## Quick Start

### Installation

```bash
pip install git+https://github.com/mutinex/mmm-eval.git
```

### Basic Usage

```bash
# Run with default settings
mmm-eval --input-data-path data.csv --framework pymc-marketing

# Run with custom configuration
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/

# Run specific tests
mmm-eval --input-data-path data.csv --framework pymc-marketing --test-names holdout_accuracy in_sample_accuracy cross_validation
```

## Documentation

The official hosted documentation can be found [here](https://mutinex.github.io/mmm-eval/).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
