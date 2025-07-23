# mmm-eval

An evaluation framework for Marketing Mix Models (MMM).

## Overview

mmm-eval provides a standardized approach for evaluation and comparison of different MMM frameworks using a full suite of validation tests including holdout accuracy, in-sample accuracy,
cross-validated holdout accuracy, refresh stability, perturbation, and placebo metrics.

## Features

- **Multi-framework support**: Evaluate PyMC-Marketing and Google Meridian models
- **Standardized metrics**: Consistent evaluation across different frameworks
- **Easy to use**: Simple CLI interface and Python API
- **Extensible**: Add new frameworks and tests easily

## Quick Start

### Installation

You can install mmm-eval directly from the repository:

```bash
pip install git+https://github.com/mutinex/mmm-eval.git
```

### Requirements

mmm-eval currently supports **Python 3.11 and 3.12** only.

If you’re building locally with Poetry, make sure you have **Poetry 2.x** installed.

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
