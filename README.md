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

Installing flash-infer for your version (eg CUDA12.8)

```sh
uv pip install flashinfer-python flashinfer-cubin
# JIT cache package (replace cu129 with your CUDA version: cu128, cu129, or cu130)
uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128
```

## Usage

See [examples/](examples/) for usage examples.


## Testing

```sh
pytest -q
```