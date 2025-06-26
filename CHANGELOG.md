# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.2] ; 2025-06-26

### Changed

- `PyMCFitSchema` now has all null defaults
- New method `fit_config_dict_without_non_provided_fields` dumps validated model to dict
  except for fields that have null values after being validated by PyMCFitSchema

## [0.4.1] ; 2025-06-26

### Changed

- Remove magic strings in `configs/`
- Remove `configs/utils.py` as unused and implemented elsewhere
- Reduce samples for test_configs

## [0.4.0] ; 2025-06-25

### Changed

- Added wrapper function `run_evaluate()` to improve workflow
- Return results as df
- `save_results()` moved outside Evaluator class

## [0.3.0] ; 2025-06-25

### Added

- Suite of 4 core validation tests
- Validation test orchestrator

## [0.2.1] ; 2025-06-25

### Changed

- Moved generic config methods in its `BaseConfig`
- Moved generic validation checks from `pymc.py` in validation class
- Added channel_columns and control_vars as properties into pymc config

## [0.2.0] ; 2025-06-25

### Changed

- Added support for various edge cases to `PyMCAdapter`
- Updated usage example notebook
- Update `PyMCAdapter` test suite

## [0.1.3] - 2025-06-24

### Changed

- Config loading class to use native pydantic for validation
- `date_column` and `channel_columns` now calculated config fields
- Changed Config to have an ABC, which pymc and other frameworks can inherit from
- Simplified logic for parsing model object and saving to json config

## [0.1.2] - 2025-06-23

### Added

- Data loading pipeline

## [0.1.1] - 2025-06-23

### Added

- Tools version
- Setup script for environment

### Fixed

- `pyproject.toml` to be Poetry-specific configuratin instead of PEP621 standard metadata

## [0.1.0] - 2025-06-16

### Added

- Initial project structure
- Core evaluation functionality
- Basic metrics: accuracy, precision, recall, f1_score
- Comprehensive test suite
- Documentation and contributing guidelines
- Poetry for dependency management

### Changed

- Migrated from setuptools to Poetry for packaging and dependency management
- Updated development workflow to use Poetry commands
- Restructured dependency management with Poetry groups

### Added

- Initial release of mmm-eval
- Core `evaluate` function for model evaluation
- Support for multiple metrics in a single call
- Binary classification metrics
- Comprehensive documentation
- MIT license
- Poetry-based project structure

### Dependencies

- numpy ^1.20.0
- pandas ^1.3.0

### Development Dependencies

- pytest ^6.0
- pytest-cov ^2.0
- black ^22.0
- isort ^5.0
- flake8 ^4.0
- mypy ^0.900
- pre-commit ^2.0

### Documentation Dependencies

- sphinx ^4.0
- sphinx-rtd-theme ^1.0
- myst-parser ^0.17
