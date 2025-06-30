# `mmm-eval`: Project Roadmap

This document outlines the consolidated plan to elevate `mmm-eval` into a best-in-class, community-friendly, and extensible open-source library. It merges the high-level OSS best practices with a detailed refactoring plan for the core evaluation logic.

---

### **1. Foundational Elements & Discoverability**

- **Comprehensive `README.md`:** Enhance the `README.md` with a compelling pitch, key features, clear examples, prominent links, and badges for CI, coverage, and PyPI status.
- **Academic Citation:** Create a `CITATION.cff` file to make the project easily citable.
- **Public Roadmap:** Maintain this `ROADMAP.md` file and link to it from the `README.md`.

---

### **2. Code Quality & Automation**

- **Consistent Style:** Implement an `.editorconfig` file and a `.pre-commit-config.yaml` with hooks for `black`, `ruff`, `isort`, and `mdformat`.
- **Conventional Commits:** Enforce Conventional Commits using `commitlint` to enable automated versioning and changelog generation.
- **Strict Typing:** Enforce strict type annotations throughout the core logic using `pyright --strict` and Ruff's `ANN*` rules. Increase overall type-hint coverage.

---

### **3. Core Logic Refactoring & Extensibility (Detailed Plan)**

This is a major initiative to refactor the core evaluation engine for flexibility and performance. Below are concrete examples of the proposed changes.

#### **3.1. Refactor `ValidationTestOrchestrator` for Extensibility**

The orchestrator will be redesigned to allow dynamic registration of tests.

**Before:** Tests are hardcoded, making the class difficult to extend.

```python
# mmm_eval/core/validation_test_orchestrator.py (Current)
class ValidationTestOrchestrator:
    def __init__(self):
        # User cannot add their own tests without modifying this list
        self.tests: dict[ValidationTestNames, type[BaseValidationTest]] = {
            ValidationTestNames.ACCURACY: AccuracyTest,
            ValidationTestNames.CROSS_VALIDATION: CrossValidationTest,
            # ...
        }

    def validate(self, ..., test_names: list[ValidationTestNames]):
        results = {}
        for test_name in test_names:
            # Inefficient: new instance created on every run
            test_instance = self.tests[test_name]()
            test_result = test_instance.run_with_error_handling(adapter, data)
            results[test_name] = test_result
        return ValidationResults(results)
```

**After:** The orchestrator accepts test instances, making it a true pluggable framework.

```python
# mmm_eval/core/validation_test_orchestrator.py (Proposed)
class ValidationTestOrchestrator:
    def __init__(self):
        self._tests: dict[str, BaseValidationTest] = {}

    def register_test(self, test: BaseValidationTest):
        """Registers a new validation test instance."""
        if test.name in self._tests:
            raise ValueError(f"Test '{test.name}' is already registered.")
        self._tests[test.name] = test

    def validate(self, ..., test_names: Iterable[str] | None = None):
        """If test_names is None, run all registered tests."""
        names_to_run = test_names if test_names is not None else self._tests.keys()

        # More Pythonic and validates all requested tests exist before running
        tests_to_run = {name: self._tests[name] for name in names_to_run}

        results = {
            name: test.run_with_error_handling(adapter, data)
            for name, test in tests_to_run.items()
        }
        return ValidationResults(results)
```

#### **3.2. Simplify the `Evaluator`**

The `Evaluator` will become a clean, high-level entry point.

**Before:** The `Evaluator` contains complex logic that is tightly coupled to the orchestrator's implementation details.

```python
# mmm_eval/core/evaluator.py (Current)
class Evaluator:
    def __init__(self, data: pd.DataFrame, test_names: tuple[str, ...] | None = None):
        self.validation_orchestrator = ValidationTestOrchestrator()
        self.data = data
        # Complex logic, breaks encapsulation by calling orchestrator's private method
        self.test_names = (
            self._get_test_names(test_names)
            if test_names else self.validation_orchestrator._get_all_test_names()
        )
    # ...
```

**After:** The `Evaluator` delegates all orchestration logic, becoming simpler and more focused.

```python
# mmm_eval/core/evaluator.py (Proposed)
class Evaluator:
    def __init__(self, data: pd.DataFrame, orchestrator: ValidationTestOrchestrator):
        self.data = data
        self.orchestrator = orchestrator

    def evaluate_framework(
        self,
        framework: str,
        config: BaseConfig,
        test_names: Iterable[str] | None = None
    ) -> ValidationResults:
        """A clean, high-level entry point."""
        adapter = get_adapter(framework, config)
        # Simply delegate to the orchestrator
        return self.orchestrator.validate(
            adapter=adapter,
            data=self.data,
            test_names=test_names,
        )
```

#### **3.3. Implement a Plugin Architecture via Entry Points**

This allows other Python packages to automatically register their own tests.

**`pyproject.toml` of an external package:**

```toml
[project.entry-points."mmm_eval.validation_tests"]
my_custom_test = "my_package.tests:MyCustomTest"
```

**Proposed addition to `mmm-eval`:**

```python
# mmm_eval/core/entry_points.py (Proposed)
import importlib.metadata

def discover_and_register_tests(orchestrator: ValidationTestOrchestrator):
    """Finds and registers tests from installed packages."""
    entry_points = importlib.metadata.entry_points(group="mmm_eval.validation_tests")
    for entry_point in entry_points:
        test_class = entry_point.load()
        orchestrator.register_test(test_class())
```

---

### **4. Documentation**

- **Automated API Reference:** Use `mkdocs` with `mkdocstrings` and `griffe` to generate a complete API reference from NumPy-style docstrings.
- **In-Depth Guides:** Expand user guides and examples, including tutorials on creating and registering custom tests.
- **Automated Workflow:**
  - Automatically build and deploy documentation to GitHub Pages.
  - Enable versioned documentation.
  - Auto-generate a changelog page using `towncrier` or a similar tool.

---

### **5. Testing & Continuous Integration**

- **Comprehensive Test Suite:**
  - Run tests against a matrix of Python versions and operating systems (Ubuntu, macOS, Windows).
  - Add end-to-end CLI tests, property-based tests with `hypothesis`, and performance regression benchmarks with `pytest-bench`.
- **Robust CI Pipeline:**
  - Implement dedicated CI jobs for linting, static typing, security scanning, and a strict documentation build.
  - Enforce a minimum test coverage threshold (e.g., 90%) and fail the build if it drops.

---

### **6. Community & Contribution**

- **GitHub Presence:**
  - Create a full suite of GitHub templates: `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, issue/PR templates.
  - Introduce a `CODEOWNERS` file for explicit review responsibilities.
  - Add a `.github/FUNDING.yml` file.
  - Enable GitHub Discussions for community interaction.
- **Acknowledge Contributors:** Use the "All Contributors" bot to recognize all forms of contribution.

---

### **7. Packaging & Distribution**

- **Automated Releases:**
  - Use `semantic-release` to automate versioning, changelog generation, and PyPI publishing based on Conventional Commits.
  - Sign Git tags and release artifacts.
- **Supply Chain Security:**
  - Build and verify both `sdist` and `wheel` artifacts in CI (run `twine check`).
  - Publish to Test PyPI in pull-request builds to validate installability before production releases.
  - Generate and attach a CycloneDX SBOM to every release.
- **Rich Metadata:** Add complete Trove classifiers, `license = {file = "LICENSE"}`, and expose `__version__` via `importlib.metadata` in `mmm_eval/__init__.py`.

---

### **8. Security**

- **Clear Policies:** Create a `SECURITY.md` file outlining the vulnerability reporting process and CVE disclosure timeline.
- **Automated Security:**
  - Configure `Dependabot` for automatic dependency updates (weekly security-update PRs).
  - Integrate the OpenSSF Scorecard and run `pip-audit` and `Bandit` scans in CI.
  - Schedule a weekly `pip-audit` job to detect newly disclosed vulnerabilities.
