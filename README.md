# A simple, extensible LLM Annotator

This repository provides a small, resumable framework for annotating datasets with
LLMs (via `vllm`). Below is a minimal usage example showing how to instantiate the
`Annotator` class and run a short annotation job.

## Installation

Recommended:

```sh
uv add llm-annotator
```

or

```sh
pip install llm-annotator
```

## Usage

See [examples/](examples/) for usage examples.


## Testing

```sh
pytest -q
```