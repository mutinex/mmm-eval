# Testing

This guide covers testing practices and procedures for the BenjaMMMin project.

## Testing Philosophy

We follow these testing principles:

- **Comprehensive coverage**: Aim for high test coverage across all modules
- **Fast feedback**: Tests should run quickly to enable rapid development
- **Reliable**: Tests should be deterministic and not flaky
- **Maintainable**: Tests should be easy to understand and modify
- **Realistic**: Tests should reflect real-world usage patterns

## Test Structure

The test suite is organized as follows:

```
tests/
├── test_adapters/               # Framework adapter tests
├── test_configs/                # Configuration object tests
├── test_core/                   # Core functionality tests
├── test_data/                   # Data handling tests
└── test_validation_tests/       # Metrics calculation tests
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
poetry run pytest

# Run tests with verbose output
poetry run pytest -v

# Run tests with coverage
poetry run pytest --cov=mmm_eval

# Run tests in parallel
poetry run pytest -n auto
```

### Running Specific Test Categories

```bash
# Run only unit tests
poetry run pytest tests/unit/

# Run only integration tests
poetry run pytest tests/integration/

# Run tests for a specific module
poetry run pytest tests/unit/test_core/

# Run tests matching a pattern
poetry run pytest -k "test_accuracy"
```

### Running Tests with Markers

```bash
# Run integration tests only
poetry run pytest -m integration

# Run slow tests only
poetry run pytest -m slow

# Skip slow tests
poetry run pytest -m "not slow"
```

## Test Types

### Unit Tests

Unit tests verify individual functions and classes in isolation. They should:

- Test one specific behavior or functionality
- Use mocks for external dependencies
- Be fast and deterministic
- Have clear, descriptive names

Example unit test:

```python
def test_calculate_mape_returns_correct_value():
    """Test that MAPE calculation returns expected results."""
    actual = [100, 200, 300]
    predicted = [110, 190, 310]
    
    result = calculate_mape(actual, predicted)
    
    expected = 10.0  # 10% average error
    assert result == pytest.approx(expected, rel=1e-2)
```

### Integration Tests

Integration tests verify that multiple components work together correctly. They:

- Test the interaction between different modules
- Use real data and minimal mocking
- May take longer to run
- Are marked with the `@pytest.mark.integration` decorator

Example integration test:

```python
@pytest.mark.integration
def test_pymc_marketing_evaluation_workflow():
    """Test complete PyMC Marketing evaluation workflow."""
    # Setup test data
    data = load_test_data()
    
    # Run evaluation
    result = evaluate_framework(
        data=data,
        framework="pymc-marketing",
        config=test_config
    )
    
    # Verify results
    assert result.accuracy > 0.8
    assert result.cross_validation_score > 0.7
    assert result.refresh_stability > 0.6
```

## Test Data and Fixtures

### Using Fixtures

Pytest fixtures provide reusable test data and setup:

```python
@pytest.fixture
def sample_mmm_data():
    """Provide sample MMM data for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'sales': np.random.normal(1000, 100, 100),
        'tv_spend': np.random.uniform(0, 1000, 100),
        'radio_spend': np.random.uniform(0, 500, 100),
        'digital_spend': np.random.uniform(0, 800, 100)
    })

def test_data_validation(sample_mmm_data):
    """Test data validation with sample data."""
    validator = DataValidator()
    result = validator.validate(sample_mmm_data)
    assert result.is_valid
```

### Test Data Management

- Use realistic but synthetic data
- Keep test data files small and focused
- Document the structure and purpose of test data

## Mocking and Stubbing

### When to Mock

Mock external dependencies to:

- Speed up tests
- Avoid network calls
- Control test conditions
- Test error scenarios

### Mocking Examples

```python
from unittest.mock import Mock, patch

def test_api_call_with_mock():
    """Test API call with mocked response."""
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {'status': 'success'}
        mock_get.return_value.status_code = 200
        
        result = fetch_data_from_api()
        
        assert result['status'] == 'success'
        mock_get.assert_called_once()
```

## Test Coverage

### Coverage Goals

- **Minimum coverage**: 80% for all modules
- **Target coverage**: 90% for critical modules
- **Critical modules**: Core evaluation logic, data validation, metrics calculation

### Coverage Reports

```bash
# Generate HTML coverage report
poetry run pytest --cov=mmm_eval --cov-report=html

# Generate XML coverage report (for CI)
poetry run pytest --cov=mmm_eval --cov-report=xml

# View coverage summary
poetry run pytest --cov=mmm_eval --cov-report=term-missing
```

### Coverage Configuration

Configure coverage in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["mmm_eval"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/__pycache__/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod"
]
```

## Performance Testing

### Benchmark Tests

For performance-critical code, use benchmark tests:

```python
def test_mape_calculation_performance(benchmark):
    """Benchmark MAPE calculation performance."""
    actual = np.random.normal(1000, 100, 10000)
    predicted = np.random.normal(1000, 100, 10000)
    
    result = benchmark(lambda: calculate_mape(actual, predicted))
    
    assert result > 0
```

### Memory Usage Tests

Monitor memory usage in tests:

```python
import psutil
import os

def test_memory_usage():
    """Test that operations don't use excessive memory."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Run memory-intensive operation
    result = process_large_dataset()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (< 100MB)
    assert memory_increase < 100 * 1024 * 1024
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:

- Every pull request
- Every push to main branch
- Scheduled runs (nightly)

### CI Configuration

The CI pipeline includes:

1. **Linting**: Code style and quality checks
2. **Type checking**: Static type analysis
3. **Unit tests**: Fast feedback on basic functionality
4. **Integration tests**: Verify component interactions
5. **Coverage reporting**: Track test coverage trends

### Pre-commit Hooks

Install pre-commit hooks to catch issues early:

```bash
# Install pre-commit
poetry add --group dev pre-commit

# Install hooks
pre-commit install

# Run all hooks
pre-commit run --all-files
```

## Debugging Tests

### Verbose Output

```bash
# Run with maximum verbosity
poetry run pytest -vvv

# Show local variables on failures
poetry run pytest -l

# Stop on first failure
poetry run pytest -x
```

### Debugging with pdb

```python
def test_debug_example():
    """Example of using pdb for debugging."""
    import pdb; pdb.set_trace()  # Breakpoint
    result = complex_calculation()
    assert result > 0
```

### Test Isolation

Ensure tests don't interfere with each other:

```python
@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    # Setup
    yield
    # Teardown
    cleanup_global_state()
```

## Best Practices

### Test Naming

- Use descriptive test names that explain the expected behavior
- Follow the pattern: `test_[function]_[scenario]_[expected_result]`
- Include edge cases and error conditions

### Test Organization

- Group related tests in classes
- Use fixtures for common setup
- Keep tests focused and single-purpose

### Assertions

- Use specific assertions (`assert result == expected`)
- Avoid complex logic in assertions
- Use appropriate assertion methods (`assertIn`, `assertRaises`, etc.)

### Test Data

- Use realistic test data
- Avoid hardcoded magic numbers
- Document test data assumptions

### Documentation

- Write clear docstrings for test functions
- Explain complex test scenarios
- Document test data sources and assumptions

## Common Pitfalls

### Flaky Tests

Avoid flaky tests by:

- Not relying on timing or external services
- Using deterministic random seeds
- Properly mocking external dependencies
- Avoiding shared state between tests

### Slow Tests

Keep tests fast by:

- Using appropriate mocks
- Minimizing I/O operations
- Using efficient test data
- Running tests in parallel when possible

### Over-Mocking

Don't over-mock:

- Test the actual behavior, not the implementation
- Mock only external dependencies
- Use real objects when possible

## Getting Help

If you encounter testing issues:

1. Check the [pytest documentation](https://docs.pytest.org/)
2. Review existing tests for examples
3. Ask questions in project discussions
4. Consult the [Contributing Guide](contributing.md) 