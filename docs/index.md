# Welcome to mmm-eval

An evaluation framework for Marketing Mix Modeling (MMM) frameworks.

## Quick Start

Get started with mmm-eval in just a few steps:

### 1. Install mmm-eval

```bash
pip install git+https://github.com/mutinex/mmm-eval.git
```

### 2. Prepare your data

Your data should include:
- Date column
- Target variable (e.g., sales, conversions)
- Media spend columns
- Revenue column (for ROI calculations)

Example data structure:
```csv
date,sales,revenue,tv_spend,digital_spend
2023-01-01,1000,7000,5000,2000
2023-01-02,1200,8000,5500,2200
```

### 3. Create a configuration

For PyMC-Marketing:
```python
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from mmm_eval.configs import PyMCConfig

model = MMM(
    date_column="date",
    channel_columns=["tv_spend", "digital_spend"],
    adstock=GeometricAdstock(l_max=4),
    saturation=LogisticSaturation()
)

config = PyMCConfig.from_model_object(
    model_object=model,
    revenue_column="revenue"
)
```

### 4. Run evaluation

```bash
mmm-eval --input-data-path data.csv --config-path config.json --output-path ./output --framework pymc-marketing
```

## Documentation

### Getting Started
- **[Installation](getting-started/installation.md)** - Get up and running with mmm-eval in minutes.
- **[Quick Start](getting-started/quick-start.md)** - Learn the basics with a hands-on example.
- **[Configuration](getting-started/configuration.md)** - Configure your MMM frameworks.

### User Guide
- **[CLI Reference](user-guide/cli.md)** - Learn how to use mmm-eval effectively.
- **[Data Requirements](user-guide/data.md)** - Understand data format and requirements.
- **[Frameworks](user-guide/frameworks.md)** - Supported MMM frameworks.
- **[Tests](user-guide/tests.md)** - Available validation tests.
- **[Metrics](user-guide/metrics.md)** - Understanding evaluation metrics.

### Examples
- **[Basic Usage](examples/basic-usage.md)** - Practical examples and use cases.

### Development
- **[Contributing](development/contributing.md)** - How to contribute to mmm-eval.
- **[Setup](development/setup.md)** - Development environment setup.
- **[Testing](development/testing.md)** - Testing practices and procedures.

## Features

### Multi-Framework Support
- **PyMC-Marketing**: Bayesian MMM framework using PyMC
- **Google Meridian**: Google's MMM framework
- **Extensible**: Easy to add new frameworks

### Extensive Testing
- **Holdout Accuracy Tests**: Out-of-sample prediction accuracy
- **In-Sample Accuracy Tests**: Model fit and overfitting detection
- **Cross-Validation**: Time series cross-validation
- **Refresh Stability**: Model stability over time
- **Perturbation Tests**: Model robustness to data changes
- **Placebo Tests**: Detection of spurious correlations

### Standardized Evaluation
- Consistent metrics across frameworks
- Reproducible results
- Industry-standard validation approaches

## Quick Navigation

### [Getting Started](getting-started/installation.md)
Get up and running with mmm-eval in minutes.

### [User Guide](user-guide/cli.md)
Learn how to use mmm-eval effectively.

### [Examples](examples/basic-usage.md)
Practical examples and use cases.

### [Development](development/contributing.md)
Contribute to mmm-eval development. 