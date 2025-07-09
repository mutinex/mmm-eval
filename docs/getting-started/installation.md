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

### Using pip

```bash
pip install git+https://github.com/Mutiny-Group/mmm-eval.git
```

**From Source:**
```bash
git clone https://github.com/Mutiny-Group/mmm-eval.git
cd mmm-eval
pip install -e .
```

### Development Installation

For development work:

```bash
git clone https://github.com/Mutiny-Group/mmm-eval.git
cd mmm-eval
poetry install
```

## Dependencies

BenjaMMMin has the following key dependencies:

- **Python 3.11+** - Required for modern Python features
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **PyMC-Marketing** - Bayesian MMM framework
- **Google Meridian** - Google's MMM framework
- **SciPy** - Scientific computing
- **PyTensor** - Computational backend
- **Pandera** - Data validation
- **Pydantic** - Data validation and settings
- **Click** - CLI framework
- **PyArrow** - Data format support

## Verification

After installation, verify that BenjaMMMin is working correctly:

```bash
# Check if BenjaMMMin is installed
python -c "import mmm_eval; print('BenjaMMMin installed successfully!')"

# Test the CLI
benjammmin --help
```

## Troubleshooting

### Common Issues

1. **Python Version**: Ensure you're using Python 3.11 or higher
2. **Dependencies**: Some dependencies may require compilation tools
3. **Virtual Environment**: Use a virtual environment to avoid conflicts

### macOS Specific

If you're on macOS 15+ (Sequoia), you may need to install the latest Xcode Command Line Tools:

```bash
xcode-select --install
```

### Virtual Environment Setup

For isolated installation:

```bash
python -m venv benjammmin-env
source benjammmin-env/bin/activate  # On Windows: benjammmin-env\Scripts\activate
pip install git+https://github.com/Mutiny-Group/mmm-eval.git
```

## Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/Mutiny-Group/mmm-eval/issues) for known problems
2. Review the [Troubleshooting Guide](troubleshooting.md)
3. Join our [Discussions](https://github.com/Mutiny-Group/mmm-eval/discussions) for community support

## Next Steps

Once you have BenjaMMMin installed, check out the [Quick Start](quick-start.md) guide to begin using it. 