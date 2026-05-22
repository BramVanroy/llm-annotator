# A simple, extensible LLM-based dataset generator and annotator

[![CI](https://github.com/BramVanroy/llm-annotator/actions/workflows/ci.yml/badge.svg)](https://github.com/BramVanroy/llm-annotator/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/BramVanroy/llm-annotator/branch/main/graph/badge.svg)](https://codecov.io/gh/BramVanroy/llm-annotator)
[![PyPI version](https://badge.fury.io/py/llm-annotator.svg)](https://badge.fury.io/py/llm-annotator)
[![Python versions](https://img.shields.io/pypi/pyversions/llm-annotator.svg)](https://pypi.org/project/llm-annotator/)
[![License](https://img.shields.io/github/license/BramVanroy/llm-annotator)](LICENSE)
![GitHub tag](https://img.shields.io/github/v/tag/BramVanroy/llm-annotator)


This repository provides a small, resumable framework for annotating datasets with LLMs (via `vllm`).

## Documentation

📚 **[Read the full documentation](https://bramvanroy.github.io/llm-annotator/)** for detailed guides, API reference, and examples.

## Installation

Recommended:

```sh
uv add llm-annotator
```

or

```sh
pip install llm-annotator
```

Installing flash-infer for your version (eg CUDA12.8)

```sh
uv pip install flashinfer-python flashinfer-cubin
# JIT cache package (replace cu129 with your CUDA version: cu128, cu129, or cu130)
uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128
```

## Usage

Quick example:

```python
from llm_annotator import Annotator

# Annotate a dataset with sentiment classification
with Annotator(model="meta-llama/Llama-3.2-3B-Instruct", max_model_len=4096) as anno:
    ds = anno.annotate_dataset(
        output_dir="outputs/sentiment",
        full_prompt_template="Classify the sentiment: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=100,
    )
```

See the **[documentation](https://bramvanroy.github.io/llm-annotator/)** for more examples, including:
- Structured output with JSON schemas
- Custom validation and postprocessing
- Large-scale streaming annotation
- Generating datasets from scratch
- Multi-GPU support

Or check out the [examples/](examples/) directory for complete working examples.


## Testing

```sh
make test
```

`make test` runs the fast suite and skips tests marked as `slow`.

Additional test targets:

```sh
# Fast tests (same as `make test`)
make test-fast

# Slow tests only
make test-slow

# Integration tests only
make test-integration

# Entire suite (fast + slow)
make test-all
```

You can also run markers directly with pytest:

```sh
uv run pytest -m "not slow"
uv run pytest -m "slow"
uv run pytest -m "integration"
```

Slow and integration tests may load local models, require more runtime, or depend on optional components.

## Building documentation

Build the documentation locally:

```sh
make docs
```

Serve the documentation locally (at http://localhost:8000):

```sh
make docs-serve
```

The documentation is automatically built and deployed to GitHub Pages when changes are pushed to the `main` branch. The pre-commit hook will check that documentation builds successfully before allowing a push if docstrings or documentation files have changed.