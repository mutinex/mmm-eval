# mmm-eval

A simple and powerful evaluation package for machine learning models.

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

```python
from mmm_eval import evaluate

# Your evaluation code here
result = evaluate(predictions, targets)
print(result)
```

## Features

- Easy-to-use evaluation metrics
- Extensible architecture
- Comprehensive documentation
- Well-tested codebase

## Development

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/mmm-eval.git
cd mmm-eval

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Running Tests
```bash
poetry run pytest
```

### Code Quality
```bash
# Format code
poetry run black src tests

# Sort imports
poetry run isort src tests

# Run linting
poetry run flake8 src tests

# Type checking
poetry run mypy src
```

## Documentation

For detailed documentation, please visit our [documentation site](https://github.com/yourusername/mmm-eval).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes. 