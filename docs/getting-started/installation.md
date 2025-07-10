# Installation

This guide will help you install mmm-eval on your system.

## Prerequisites

mmm-eval requires Python 3.11 or higher. Make sure you have Python installed on your system.

## Installation Methods

### Using Poetry (Recommended)

The recommended way to install mmm-eval is using Poetry. Ensure you're using Poetry
`2.x.x`.

```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Add mmm-eval to your project
poetry add git+https://github.com/mutinex/mmm-eval.git
```

### Using pip

```bash
pip install git+https://github.com/mutinex/mmm-eval.git
```

### From Source

```bash
# Clone the repository
git clone https://github.com/mutinex/mmm-eval.git
cd mmm-eval

# Install using Poetry
poetry install
```

## Dependencies

mmm-eval has the following key dependencies:

### Core Dependencies
- **PyMC-Marketing**: For PyMC-Marketing framework support
- **Meridian**: For Google Meridian framework support
- **Pandas**: For data manipulation
- **NumPy**: For numerical computations
- **SciPy**: For scientific computations

### Optional Dependencies
- **Matplotlib**: For plotting (optional)
- **Seaborn**: For enhanced plotting (optional)

## Verification

After installation, verify that mmm-eval is working correctly:

```bash
# Check if mmm-eval is installed
python -c "import mmm_eval; print(f'mmm-eval version: {mmm_eval.__version__}')"

# Test the CLI
mmm-eval --help
```

## Development Setup

If you want to contribute to mmm-eval, set up a development environment:

### Using Poetry

```bash
# Clone the repository
git clone https://github.com/mutinex/mmm-eval.git
cd mmm-eval

# Install dependencies
poetry install

# Activate the environment
poetry shell

# Test the installation
poetry run mmm-eval --help
```

### Using Virtual Environment

```bash
# Create virtual environment
python -m venv mmm-eval-env
source mmm-eval-env/bin/activate  # On Windows: mmm-eval-env\Scripts\activate

# Install in development mode
pip install -e .

# Test the installation
mmm-eval --help
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're using Python 3.11+
2. **CLI not found**: Ensure the package is installed correctly
3. **Dependency conflicts**: Use a virtual environment

### Getting Help

If you encounter installation issues:

- Check the [GitHub Issues](https://github.com/mutinex/mmm-eval/issues)
- Join our [Discussions](https://github.com/mutinex/mmm-eval/discussions)

## Next Steps

Once you have mmm-eval installed, check out the [Quick Start](quick-start.md) guide to begin using it. 