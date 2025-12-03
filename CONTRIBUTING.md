# Contributing to llm-annotator

Thank you for your interest in contributing to llm-annotator! This document provides guidelines and instructions for contributing.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/BramVanroy/llm-annotator.git
cd llm-annotator
```

2. Set up the development environment:
```bash
make setup
```

This will:
- Install all development dependencies
- Set up pre-commit hooks

## Making Changes

### Code Changes

1. Create a new branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes to the code

3. Ensure code quality:
```bash
make quality  # Check code quality
make style    # Auto-format code
```

4. Run tests:
```bash
pytest -q
```

### Documentation Changes

When you modify docstrings in the source code or documentation files, the documentation will be automatically validated before you can push your changes.

#### Writing Docstrings

- Use Google-style docstrings
- Include comprehensive examples in docstrings for public methods
- Examples should be realistic and runnable (even if they require resources)
- Document all parameters, return values, and exceptions

Example:
```python
def my_function(param1: str, param2: int = 10) -> bool:
    """Short description of the function.

    Longer description with more details about what the function does,
    how it works, and any important notes.

    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 10.

    Returns:
        Description of return value.

    Examples:
        Basic usage:

        >>> result = my_function("test", 5)
        >>> print(result)
        True

        Advanced usage:

        >>> result = my_function("test")
        >>> print(result)
        False
    """
    return len(param1) > param2
```

#### Building Documentation Locally

Build the documentation:
```bash
make docs
```

The output will be in `docs/_build/html/`.

Serve the documentation locally:
```bash
make docs-serve
```

Then open http://localhost:8000 in your browser.

#### Documentation Structure

- `docs/index.md` - Main landing page
- `docs/getting-started.md` - Tutorial for new users
- `docs/api-reference.md` - Auto-generated API documentation from docstrings
- `docs/examples.md` - Practical examples and use cases
- `docs/conf.py` - Sphinx configuration

## Pre-commit Hooks

Pre-commit hooks will automatically run before you push:

1. **Code Quality** (`make quality`) - Checks code style and linting
2. **Documentation Build** - Validates that documentation builds successfully if you've modified docstrings or docs files

If the pre-commit hooks fail:
- Fix the reported issues
- Stage your fixes: `git add .`
- Try committing/pushing again

To bypass hooks (not recommended):
```bash
git push --no-verify
```

## Pull Request Process

1. Update documentation if you've changed functionality
2. Add examples to docstrings for new public methods
3. Ensure all tests pass
4. Ensure documentation builds successfully
5. Update the README.md if needed
6. Create a pull request with a clear description of changes

## Documentation Deployment

Documentation is automatically deployed to GitHub Pages when changes are merged to `main`:

1. GitHub Actions builds the documentation
2. Deploys to https://bramvanroy.github.io/llm-annotator/
3. Usually available within a few minutes

## Questions?

If you have questions, please:
- Check existing issues and discussions
- Open a new issue for bugs or feature requests
- Start a discussion for questions

Thank you for contributing!
