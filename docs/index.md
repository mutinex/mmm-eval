# Welcome to BenjaMMMin

An open-source evaluation framework for Marketing Mix Models (MMMs).

## Quick Start

Get started with BenjaMMMin in just a few steps:

### Installation

**Using Poetry (Recommended):**
```bash
poetry add git+https://github.com/Mutiny-Group/mmm-eval.git
```

**From Source:**
```bash
git clone https://github.com/Mutiny-Group/mmm-eval.git
cd mmm-eval
poetry install
```

**Using pip:**
```bash
pip install git+https://github.com/Mutiny-Group/mmm-eval.git
```

**From Source:**
```bash
git clone https://github.com/Mutiny-Group/mmm-eval.git
cd mmm-eval
pip install -e .
```

### Basic Usage

```bash
benjammmin --input-data-path data.csv --framework pymc-marketing
```

> **Note**: BenjaMMMin is currently in development. For production use, we recommend installing from the latest release tag:
> 
> ```bash
> poetry add git+https://github.com/Mutiny-Group/mmm-eval.git@v0.4.2
> ```

## Documentation

### Getting Started
- **[Installation](getting-started/installation.md)** - How to install BenjaMMMin
- **[Quick Start](getting-started/quick-start.md)** - Get up and running quickly
- **[Configuration](getting-started/configuration.md)** - Configure your evaluation

### User Guide
- **[Command Line Interface](user-guide/cli.md)** - Using the CLI
- **[Frameworks](user-guide/frameworks.md)** - Supported MMM frameworks
- **[Tests](user-guide/tests.md)** - Available validation tests
- **[Metrics](user-guide/metrics.md)** - Understanding evaluation metrics
- **[Data](user-guide/data.md)** - Data format requirements

### API Reference
- **[Adapters](api/adapters.md)** - Framework adapters
- **[Core](api/core.md)** - Core evaluation functionality
- **[CLI](api/cli.md)** - Command line interface
- **[Data](api/data.md)** - Data processing utilities
- **[Metrics](api/metrics.md)** - Evaluation metrics

### Examples
- **[Basic Usage](examples/basic-usage.md)** - Simple evaluation examples
- **[Custom Configuration](examples/custom-configuration.md)** - Advanced configuration
- **[Advanced Scenarios](examples/advanced-scenarios.md)** - Complex use cases

### Development
- **[Contributing](development/contributing.md)** - How to contribute to BenjaMMMin
- **[Development Setup](development/setup.md)** - Setting up development environment
- **[Testing](development/testing.md)** - Testing guidelines

## Features

- **Multi-framework support** - Evaluate PyMC-Marketing, Google Meridian, and other MMM frameworks
- **Comprehensive validation tests** - Accuracy, cross-validation, refresh stability, and perturbation tests
- **Standardized metrics** - MAPE, RMSE, R-squared, and other industry-standard metrics
- **Flexible data handling** - Support for custom column names and data formats
- **CLI interface** - Easy-to-use command-line tool for evaluation

## Quick Navigation

- **[Getting Started](getting-started/installation.md)** - Get up and running with BenjaMMMin in minutes.
- **[User Guide](user-guide/cli.md)** - Learn how to use BenjaMMMin effectively.
- **[API Reference](api/core.md)** - Detailed API documentation.
- **[Examples](examples/basic-usage.md)** - Practical examples and use cases.

## 📊 Supported Frameworks

Currently supported MMM frameworks:

- **PyMC-Marketing**: Bayesian MMM framework using PyMC
- *More frameworks coming soon...*

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details on how to get started.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [License](about/license.md) file for details.

---

<div class="grid cards" markdown>

-   :fontawesome-solid-rocket: __[Quick Start](getting-started/quick-start.md)__

    Get up and running with BenjaMMMin in minutes.

-   :fontawesome-solid-book: __[User Guide](user-guide/cli.md)__

    Learn how to use BenjaMMMin effectively.

-   :fontawesome-solid-code: __[API Reference](api/core.md)__

    Explore the complete API documentation.

-   :fontawesome-solid-lightbulb: __[Examples](examples/basic-usage.md)__

    See practical examples and use cases.

</div> 