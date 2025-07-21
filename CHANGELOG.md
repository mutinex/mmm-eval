# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.1] ; 2025-07-21

### Fixed

- Package versioning logic when installing via `pip`

## [0.11.0] ; 2025-07-21

### Added

- Placebo/falsification test

## [0.10.0] ; 2025-07-17

### Added

- New in-sample accuracy validation test, using same metrics as those used for holdout accuracy

## [0.9.1] ; 2025-07-17

### Changed

- Track Meridian testing file in repository instead of requiring user to track it

## [0.9.0] ; 2025-07-17

### Added

- SMAPE metric to accuracy and CV accuracy tests

### Changed

- All metrics with "percentage" in the name to be actual percentages, e.g. 15 instead of 0.15

## [0.8.0] ; 2025-07-17

### Changed

- Changed holdout split strategy for accuracy test to fixed test set size

## [0.7.1] ; 2025-07-15

### Fixed

- Minor bugs in Meridian adapter supporting national model

## [0.7.0] ; 2025-07-10

### Changed

- Updated documentation to reflect Meridian support

## [0.6.1] ; 2025-07-09

### Added

- Support perturbation tests if non-spend media metrics are provided to Meridian

## [0.6.1] ; 2025-07-09

### Added

- Support perturbation tests if non-spend media metrics are provided to Meridian

## [0.6.0] ; 2025-07-07

### Added

- Support for Meridian via SDK only

## [0.5.0] ; 2025-07-03

### Changed

- Timeseries splitting logic to be robust to the presence of other index columns
- Added `fit_and_predict()` method to adapter class

## [0.4.5] ; 2025-06-27

### Changed

- Remove all `to_dict()`
- Added in new `to_df()`
- Passed values now contained within the test df for each metric individually

## [0.4.4] ; 2025-06-27

### Changed

- Derive `predict_kwargs` from `fit_kwargs` to control progress bar and seed for predict

## [0.4.3] ; 2025-06-27

### Changed

- `PyMCFitSchema` now is optional for user

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
