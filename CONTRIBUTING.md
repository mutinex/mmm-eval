# Contributing to BenjaMMMin

We welcome contributions to BenjaMMMin! This document provides guidelines for contributing to the project.

## Development Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management. Make sure you have Poetry installed on your system.

### Installation

If you don't have Poetry installed, you can install it by following the [official installation guide](https://python-poetry.org/docs/#installation).

### Setup Process

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/mmm-eval.git
   cd mmm-eval
   ```

3. Install the project and its dependencies:
   ```bash
   poetry install
   ```

4. Activate the virtual environment:
   ```bash
   poetry shell
   ```

5. Install pre-commit hooks:
   ```bash
   poetry run pre-commit install
   ```

## Running Tests

Run the test suite:
```bash
poetry run pytest
```

Run tests with coverage:
```bash
poetry run pytest --cov=mmm_eval --cov-report=html
```

## Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:
```bash
poetry run black src tests
poetry run isort src tests
poetry run flake8 src tests
poetry run mypy src
```

Or run all checks at once with pre-commit:
```bash
poetry run pre-commit run --all-files
```

## Adding Dependencies

### Production Dependencies
```bash
poetry add package-name
```

### Development Dependencies
```bash
poetry add --group dev package-name
```

### Documentation Dependencies
```bash
poetry add --group docs package-name
```

## Submitting Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and write tests
3. Run the test suite and ensure all tests pass
4. Run the code quality checks
5. Commit your changes with a clear commit message
6. Push to your fork and submit a pull request

## Pull Request Guidelines

- Include tests for new functionality
- Update documentation as needed
- Follow the existing code style
- Write clear, concise commit messages
- Include a description of what your PR does

## Reporting Issues

When reporting issues, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Your environment (Python version, OS, Poetry version, etc.)

## Adding New Metrics

To add a new evaluation metric:

1. Add the metric function to `src/mmm_eval/metrics.py`
2. Add the metric to the `AVAILABLE_METRICS` dictionary
3. Export the metric in `src/mmm_eval/__init__.py`
4. Write comprehensive tests in `tests/test_metrics.py`
5. Update the README if needed

## Project Structure

The project follows a standard Poetry layout:

```
benjammmin/
├── src/mmm_eval/          # Source code
├── tests/                 # Test suite
├── examples/              # Usage examples
├── pyproject.toml         # Poetry configuration and dependencies
├── README.md             # Project documentation
└── ...
```

## Questions?

If you have questions about contributing, please open an issue or start a discussion on GitHub.

Thank you for contributing to BenjaMMMin! 