# Contributing to OrganoidEnv

Thank you for your interest in contributing to **OrganoidEnv**! We welcome contributions from computational neuroscientists, neuromorphic engineers, machine learning researchers, and open-source enthusiasts.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md) (Contributor Covenant v2.1).

---

## Development Setup

### 1. Fork and Clone
```bash
git clone https://github.com/<your-username>/Organoid-Intelligence-gym-.git
cd Organoid-Intelligence-gym-
```

### 2. Set Up a Virtual Environment
```bash
# Using standard venv (Python 3.9+)
python -m venv .venv

# Activate environment:
# On Linux/macOS:
source .venv/bin/activate
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### 3. Install in Editable Mode with Dev Dependencies
```bash
pip install -e ".[dev]"
```

---

## Running Tests

Before submitting changes, ensure all tests pass:

```bash
# Run standard test suite
python -m unittest discover -s organoid_rl/tests -p "test_*.py"

# Run Brian2 sanity check
python organoid_rl/tests/sanity_check.py

# Run quick simulation step test
python organoid_rl/tests/quick_test.py

# Run diagnostic script
python organoid_rl/tests/diagnostic.py
```

If you have `pytest` installed:
```bash
pytest
```

---

## Coding Guidelines

- **Style:** We follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- **Type Annotations:** Use Python type hints where appropriate.
- **Documentation:** All public functions, classes, and environment parameters must include descriptive docstrings.
- **Deterministic Experiments:** Always provide a `seed` argument to ensure reproducibility.
- **Preserve Biological Invariants:** Avoid introducing non-local backpropagation gradients into the internal SNN dynamics; optimization should occur via sensory stimulation or local plasticity rules.

---

## Submitting Pull Requests

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with clear, descriptive commit messages:
   ```bash
   git commit -m "feat(environment): add adaptive homeostatic threshold"
   ```
3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
4. Open a Pull Request on GitHub using the PR template.
5. Ensure all CI checks pass.

---

## Reporting Issues

- **Bug Reports:** Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md). Include OS, Python version, Brian2 compiler backend, and a minimal reproducible script.
- **Feature Requests:** Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md).

---

## Questions & Discussions

Feel free to open a GitHub Discussion or reach out to the project maintainers via GitHub Issues.
