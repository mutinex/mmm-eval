# Installation

This guide will help you install BenjaMMMin on your system.

## Prerequisites

BenjaMMMin requires Python 3.11 or higher. Make sure you have Python installed on your system.

## Installation Methods

### Using Poetry (Recommended)

The recommended way to install BenjaMMMin is using Poetry:

```bash
poetry add git+https://github.com/Mutiny-Group/mmm-eval.git
```

**From Source:**
```bash
git clone https://github.com/Mutiny-Group/mmm-eval.git
cd mmm-eval
poetry install
```

**Prerequisite**: Poetry 2.x.x or later is required.

### Using pip

If you prefer using pip directly:

```bash
pip install git+https://github.com/Mutiny-Group/mmm-eval.git
```

**From Source:**
```bash
git clone https://github.com/Mutiny-Group/mmm-eval.git
cd mmm-eval
pip install -e .
```

### From Source

To install from the latest development version:

```bash
git clone https://github.com/Mutiny-Group/mmm-eval.git
cd mmm-eval
pip install -e .
```

### Development Installation

For development work, install with all development dependencies:

```bash
git clone https://github.com/Mutiny-Group/mmm-eval.git
cd mmm-eval
poetry install --with dev
```

## Dependencies

BenjaMMMin has the following key dependencies:

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

After installation, verify that BenjaMMMin is working correctly:

```bash
# Check if BenjaMMMin is installed
python -c "import mmm_eval; print(f'BenjaMMMin version: {mmm_eval.__version__}')"

# Check CLI availability
benjammmin --help
```

## Troubleshooting

### Common Issues

1. **Import Error**: If you get import errors, make sure you're using Python 3.11 or higher.

2. **Poetry Installation Issues**: If you encounter issues with Poetry:
   ```bash
   # Update Poetry to the latest version
   poetry self update
   
   # Clear Poetry cache
   poetry cache clear --all pypi
   ```

3. **Permission Errors**: On some systems, you might need to use `pip install --user` to install without admin privileges.

4. **Version Conflicts**: If you encounter dependency conflicts:
   ```bash
   # With Poetry
   poetry update
   
   # With pip (in a virtual environment)
   python -m venv benjammmin-env
   source benjammmin-env/bin/activate  # On Windows: benjammmin-env\Scripts\activate
   pip install git+https://github.com/Mutiny-Group/mmm-eval.git
   ```

### Getting Help

If you encounter any issues during installation, please:

1. Check the [GitHub Issues](https://github.com/Mutiny-Group/mmm-eval/issues) for known problems
2. Create a new issue with details about your system and the error message
3. Join our [Discussions](https://github.com/Mutiny-Group/mmm-eval/discussions) for community support

## Next Steps

Once you have BenjaMMMin installed, check out the [Quick Start](quick-start.md) guide to begin using it. 