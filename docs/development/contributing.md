# Contributing to mmm-eval

We welcome contributions from the community! This guide will help you get started with contributing to mmm-eval

## Getting Started

### Prerequisites

1. **Python 3.11+** - Required for development
2. **Git** - For version control
3. **Poetry** - For dependency management (recommended)

### Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mmm-eval.git
   cd mmm-eval
   ```

3. **Set up the development environment**:
   ```bash
   # Install dependencies
   poetry install
   
   # Activate the environment
   poetry shell
   ```

4. **(Optional) Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write your code following the [coding standards](#coding-standards)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
tox

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run linting and formatting
black mmm_eval tests
isort mmm_eval tests
ruff check mmm_eval tests
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

Where possible, please follow the [conventional commits](https://www.conventionalcommits.org/) format:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Maintenance tasks

### 5. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Coding Standards

### Python Code Style

We use several tools to maintain code quality:

- **Black** - Code formatting
- **isort** - Import sorting
- **Ruff** - Linting and additional checks
- **Pyright** - Type checking

### Code Formatting

```bash
# Format code
black mmm_eval tests

# Sort imports
isort mmm_eval tests

# Run linting
ruff check mmm_eval tests
ruff check --fix mmm_eval tests
```

### Type Hints

We use type hints throughout the codebase:

```python
from typing import List, Optional, Dict, Any

def process_data(data: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Process the input data and return results."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_mape(actual: List[float], predicted: List[float]) -> float:
    """Calculate Mean Absolute Percentage Error.
    
    Args:
        actual: List of actual values
        predicted: List of predicted values
        
    Returns:
        MAPE value as a float
        
    Raises:
        ValueError: If inputs are empty or have different lengths
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mmm_eval

# Run specific test file
pytest tests/test_metrics.py

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Use fixtures for common test data

Example test:

```python
import pytest
from mmm_eval.metrics import calculate_mape

def test_calculate_mape_basic():
    """Test basic MAPE calculation."""
    actual = [100, 200, 300]
    predicted = [110, 190, 310]
    
    mape = calculate_mape(actual, predicted)
    
    assert isinstance(mape, float)
    assert mape > 0

def test_calculate_mape_empty_input():
    """Test MAPE calculation with empty input."""
    with pytest.raises(ValueError):
        calculate_mape([], [])
```

## Documentation

### Updating Documentation

1. **Update docstrings** in the code
2. **Update markdown files** in the `docs/` directory
3. **Build and test documentation**:
   ```bash
   mkdocs serve
   ```

### Documentation Standards

- Use clear, concise language
- Include code examples
- Keep documentation up to date with code changes
- Use proper markdown formatting

## Pull Request Guidelines

### Before Submitting

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Follow coding standards**
5. **Update CHANGELOG.md**

### Pull Request Template

Use the provided pull request template and fill in all sections:

- **Description** - What does this PR do?
- **Type of change** - Bug fix, feature, documentation, etc.
- **Testing** - How was this tested?

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Documentation review** if needed
4. **Merge** after approval

## Issue Reporting

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check documentation** for solutions
3. **Update to the latest version** of mmm-eval

### Issue Template

Be sure to include:

- **Description** - Clear description of the problem
- **Steps to reproduce** - Detailed steps
- **Expected behavior** - What should happen
- **Actual behavior** - What actually happens
- **Environment** - OS, Python version, mmm-eval version
- **Additional context** - Any other relevant information

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and inclusive
- Use welcoming and inclusive language
- Be collaborative and constructive
- Focus on what is best for the community

### Communication

- **GitHub Issues** - For bug reports and feature requests
- **GitHub Discussions** - For questions and general discussion
- **Pull Requests** - For code contributions

## Getting Help

If you need help with contributing:

1. **Check the documentation** first
2. **Search existing issues** and discussions
3. **Create a new discussion** for questions
4. **Join our community** channels

## Recognition

Contributors will be recognized in:

- **README.md** - For significant contributions
- **CHANGELOG.md** - For all contributions
- **GitHub contributors** page

Thank you for contributing to mmm-eval! 🎉 