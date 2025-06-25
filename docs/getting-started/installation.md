# Installation

This guide will help you install mmm-eval on your system.

## Prerequisites

mmm-eval requires Python 3.11 or higher. Make sure you have Python installed on your system.

## Installation Methods

### Using pip (Recommended)

The easiest way to install mmm-eval is using pip:

```bash
pip install mmm-eval
```

### Using Poetry

If you're using Poetry for dependency management:

```bash
poetry add mmm-eval
```

### From Source

To install from the latest development version:

```bash
git clone https://github.com/mutinex/mmm-eval.git
cd mmm-eval
pip install -e .
```

### Development Installation

For development work, install with all development dependencies:

```bash
git clone https://github.com/mutinex/mmm-eval.git
cd mmm-eval
pip install -e ".[dev]"
```

## Dependencies

mmm-eval has the following key dependencies:

- **numpy** (>=1.17) - Numerical computing
- **pandas** (^2.0.0) - Data manipulation
- **google-meridian** (^1.1.0) - Google's Meridian MMM framework
- **pymc-marketing** (^0.14.0) - PyMC-based MMM framework
- **scipy** (>=1.13.1,<2.0.0) - Scientific computing
- **pytensor** (^2.18.0) - Tensor operations
- **pandera** (^0.24.0) - Data validation
- **pydantic** (^2.5) - Data validation and settings
- **click** (^8.1.7) - Command line interface
- **pyarrow** (^20.0.0) - Fast data interchange

## Verification

After installation, verify that mmm-eval is working correctly:

```bash
# Check if mmm-eval is installed
python -c "import mmm_eval; print(mmm_eval.__version__)"

# Check CLI availability
mmm-eval --help
```

## Troubleshooting

### Common Issues

1. **Import Error**: If you get import errors, make sure you're using Python 3.11 or higher.

2. **Permission Errors**: On some systems, you might need to use `pip install --user mmm-eval` to install without admin privileges.

3. **Version Conflicts**: If you encounter dependency conflicts, consider using a virtual environment:

```bash
python -m venv mmm-eval-env
source mmm-eval-env/bin/activate  # On Windows: mmm-eval-env\Scripts\activate
pip install mmm-eval
```

### Getting Help

If you encounter any issues during installation, please:

1. Check the [GitHub Issues](https://github.com/mutinex/mmm-eval/issues) for known problems
2. Create a new issue with details about your system and the error message
3. Join our [Discussions](https://github.com/mutinex/mmm-eval/discussions) for community support

## Next Steps

Once you have mmm-eval installed, check out the [Quick Start](quick-start.md) guide to begin using it. 