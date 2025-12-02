# A simple, extensible LLM Annotator

This repository provides a small, resumable framework for annotating datasets with
LLMs (via `vllm`).

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
pytest -q
```