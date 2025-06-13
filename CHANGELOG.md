# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [0.1.0] - 2024-01-XX

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