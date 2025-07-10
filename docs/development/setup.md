# Development Setup

This guide will help you set up a development environment for contributing to mmm-eval.

## Prerequisites

Before setting up the development environment, ensure you have the following installed:

- **Python 3.11+**: The minimum supported Python version
- **Poetry 2.x.x**: For dependency management and packaging
- **Git**: For version control

## Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mutinex/mmm-eval.git
   cd mmm-eval
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Verify the installation**:
   ```bash
   poetry run mmm-eval --help
   ```

## Development Environment

### Using Poetry (Recommended)

Poetry automatically creates and manages a virtual environment for the project:

```bash
# Activate the virtual environment
poetry shell

# Run commands within the environment
poetry run python -m pytest

# Install additional development dependencies
poetry add --group dev package-name
```

### Using pip (Alternative)

If you prefer using pip directly:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt  # If available
```

## Project Structure

```
mmm-eval/
├── mmm_eval/              # Main package
│   ├── adapters/          # Framework adapters
│   ├── cli/               # Command-line interface
│   ├── core/              # Core evaluation logic
│   ├── data/              # Data handling and validation
│   └── metrics/           # Evaluation metrics
├── tests/                 # Test suite
├── docs/                  # Documentation
├── pyproject.toml         # Project configuration
└── README.md             # Project overview
```

## Code Quality Tools

The project uses several tools to maintain code quality:

### Code Formatting

The project uses Black for code formatting:

```bash
# Format all Python files
poetry run black .

# Check formatting without making changes
poetry run black --check .
```

### Import Sorting

isort is used to organize imports:

```bash
# Sort imports
poetry run isort .

# Check import sorting
poetry run isort --check-only .
```

### Linting

Multiple linters are configured:

```bash
# Run all linters
poetry run ruff check .
poetry run flake8 .

# Auto-fix issues where possible
poetry run ruff check --fix .
```

### Type Checking

Pyright is used for static type checking:

```bash
# Run type checker
poetry run pyright
```

### (Optional) Pre-commit Hooks 

Install pre-commit hooks to automatically format and lint your code:

```bash
# Install pre-commit
poetry add --group dev pre-commit

# Install the git hook scripts
pre-commit install

# Run against all files
pre-commit run --all-files
```

## Running Tests

### Unit Tests

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=mmm_eval

# Run tests in parallel
poetry run pytest -n auto

# Run specific test file
poetry run pytest tests/test_core.py
```

### Test Coverage

```bash
# Generate coverage report
poetry run pytest --cov=mmm_eval --cov-report=html

# View coverage report
open htmlcov/index.html  # On macOS
# or
start htmlcov/index.html  # On Windows
```

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
poetry install --with docs

# Build documentation
poetry run mkdocs build

# Serve documentation locally
poetry run mkdocs serve
```

The documentation will be available at `http://localhost:8000`.

### API Documentation

API documentation is automatically generated from docstrings using `mkdocstrings`. To update the API docs:

1. Ensure your code has proper docstrings
2. Build the documentation: `poetry run mkdocs build`
3. The API reference will be updated automatically

## IDE Configuration

### VS Code

Recommended VS Code extensions:

- Python
- Pylance
- Black Formatter
- isort
- Ruff

Add to your `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.ruffEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### PyCharm

1. Open the project in PyCharm
2. Configure the Python interpreter to use the Poetry virtual environment
3. Enable auto-import organization
4. Configure Black as the code formatter

## Common Issues

### Poetry Installation Issues

If you encounter issues with Poetry:

```bash
# Update Poetry to the latest version
poetry self update

# Clear Poetry cache
poetry cache clear --all pypi
```

### Dependency Conflicts

If you encounter dependency conflicts:

```bash
# Update all dependencies
poetry update

# Remove and reinstall dependencies
poetry lock --no-update
poetry install
```

### Test Failures

If tests are failing:

1. Ensure you're using the correct Python version (3.11+)
2. Check that all dependencies are installed: `poetry install`
3. Run tests with verbose output: `poetry run pytest -v`
4. Check for any environment-specific issues

## Next Steps

Once your development environment is set up:

1. Read the [Contributing Guide](contributing.md) for guidelines
2. Check the [Testing Guide](testing.md) for testing practices
3. Look at existing issues and pull requests
4. Start with a small contribution to get familiar with the codebase

## Getting Help

If you encounter issues during setup:

1. Check the [GitHub Issues](https://github.com/mutinex/mmm-eval/issues)
2. Review the [Contributing Guide](contributing.md)
3. Ask questions in the project discussions 