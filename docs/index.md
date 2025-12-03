# llm-annotator Documentation

Welcome to the **llm-annotator** documentation! This library provides a simple, extensible framework for annotating datasets with Large Language Models (LLMs) using vLLM.

## Quick Links

```{toctree}
:maxdepth: 2
:caption: Contents

getting-started
api-reference
examples
```

## Overview

The `llm-annotator` package offers:

- **Easy dataset annotation**: Apply LLM-based annotations to existing datasets
- **Data generation from scratch**: Generate new datasets using LLM completions
- **Structured outputs**: Constrain outputs using JSON schemas
- **Resumable jobs**: Automatically resume interrupted annotation jobs
- **Streaming support**: Process large datasets efficiently
- **Hub integration**: Upload results directly to Hugging Face Hub
- **Validation & retries**: Custom validation with automatic retries for invalid outputs

## Installation

Install using pip or uv:

```bash
pip install llm-annotator
```

or

```bash
uv add llm-annotator
```

For flash-attention support (recommended for better performance):

```bash
# Replace cu128 with your CUDA version: cu128, cu129, or cu130
uv pip install flashinfer-python flashinfer-cubin
uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128
```

## Quick Start

Here's a minimal example to get started:

```python
from llm_annotator import Annotator

# Create an annotator with your chosen model
with Annotator(model="meta-llama/Llama-3.2-3B-Instruct", max_model_len=4096) as anno:
    # Annotate a dataset
    ds = anno.annotate_dataset(
        output_dir="outputs/sentiment",
        full_prompt_template="Classify the sentiment of this review: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=100,
    )
    
    print(ds)
```

## Main Components

### Annotator Class

The {class}`~llm_annotator.Annotator` class is the core of the library. It provides two main public methods:

- {meth}`~llm_annotator.Annotator.annotate_dataset`: Annotate an existing dataset
- {meth}`~llm_annotator.Annotator.generate_dataset`: Generate a new dataset from scratch

See the {doc}`api-reference` for detailed documentation of all methods and parameters.

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
