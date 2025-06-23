# mmm-eval

An open-source tool for evaluating Marketing Mix Modeling (MMM) frameworks.

## Overview

mmm-eval provides a standardized way to evaluate and compare different MMM frameworks using a comprehensive suite of validation tests including accuracy, cross-validation, refresh stability, and perturbation tests.

## Features

- **Multi-framework support**: Evaluate , PyMC-Marketing, and other MMM frameworks
- **Comprehensive validation tests**: Accuracy, cross-validation, refresh stability, and perturbation tests
- **Standardized metrics**: MAPE, RMSE, R-squared, and other industry-standard metrics
- **Flexible data handling**: Support for custom column names and data formats
- **CLI interface**: Easy-to-use command-line tool for evaluation

## Installation

### Using pip

```bash
pip install mmm-eval
```

### Using Poetry

```bash
poetry add mmm-eval
```

## Quick Start

### Command Line Interface

```bash
# Basic evaluation
mmm-eval --input-data-path data.csv --framework pymc-marketing

# With custom configuration
mmm-eval --input-data-path data.csv --framework pymc-marketing --config-path config.json --output-path results/

# Run specific tests only
mmm-eval --input-data-path data.csv --framework pymc-marketing --test-names accuracy cross_validation
```

## Development Setup

This project uses [asdf](https://asdf-vm.com/) for Python version management and [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites

1. **Install asdf**:

   ```bash
   brew install asdf
   ```

1. **Add asdf to your shell config** (add to `~/.zshrc` or `~/.bashrc`):

   ```bash
   . "$HOME/.asdf/asdf.sh"
   . "$HOME/.asdf/completions/asdf.zsh"  # for zsh
   ```

1. **Restart your terminal** or reload your shell config:

   ```bash
   source ~/.zshrc
   ```

1. **Install asdf Python plugin**:

   ```bash
   asdf plugin add python
   ```

### Quick Setup

Run the automated setup script:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

**Note:** The `chmod +x` step is required because this is a shared repository and file permissions aren't preserved in git.

### Manual Setup

If you prefer to set up manually:

1. **Install Python version** (specified in `.tool-versions`):

   ```bash
   asdf install
   ```

1. **Install Poetry** (if not already installed):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   poetry self add poetry-plugin-shell
   ```

1. **Configure Poetry to use asdf Python**:

   ```bash
   poetry env use $(asdf which python)
   ```

1. **Install dependencies**:

   ```bash
   poetry install
   ```

## Development Workflow

### Activate the environment

```bash
poetry shell
```

### Run tests and code quality

```bash
tox
```

## Supported Frameworks

- **PyMC-Marketing**: Bayesian MMM framework using PyMC

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes.
