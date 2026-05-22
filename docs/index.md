# LLM Annotator

LLM Annotator is a Python library for robust, resumable annotation and
generation workflows powered by large language models.

It provides a common interface for multiple providers:

- `VLLMOfflineClient` for local vLLM inference.
- `VLLMClient` for vLLM server endpoints.
- `OpenAIClient` for OpenAI-compatible APIs.
- `ClaudeClient` for Anthropic APIs.
- `GeminiClient` for Gemini APIs.

Provider setup details, extras, and auth variables are listed on
[Provider setup](provider-info.md).

## Install

With uv:

```bash
uv add llm-annotator
```

With pip:

```bash
pip install llm-annotator
```

Install provider extras when needed:

```bash
uv add "llm-annotator[vllm]"
uv add "llm-annotator[openai]"
uv add "llm-annotator[anthropic]"
uv add "llm-annotator[gemini]"
```

## Quickstart

Annotate a dataset:

```python
from llm_annotator import Annotator, VLLMOfflineClient

client = VLLMOfflineClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=4096,
)

with Annotator(client=client) as anno:
    ds = anno.annotate_dataset(
        output_dir="outputs/imdb-sentiment",
        prompt_template="Classify the sentiment: {text}",
        dataset_name="stanfordnlp/imdb",
        dataset_split="test",
        max_num_samples=100,
    )
```

Generate a dataset from prompts:

```python
from llm_annotator import Annotator, OpenAIClient

client = OpenAIClient(model="gpt-4o-mini")

with Annotator(client=client) as anno:
    ds = anno.generate_dataset(
        output_dir="outputs/generated",
        prompts="Create one short NER training sentence.",
        max_num_samples=50,
    )
```

## Why use it

- Resume interrupted runs from JSONL checkpoints.
- Validate and post-process outputs with custom callables.
- Enforce structured responses through JSON schemas.
- Upload incrementally to the Hugging Face Hub.

## Development

```bash
git clone https://github.com/BramVanroy/llm-annotator.git
cd llm-annotator
uv sync --dev
```

Run checks:

```bash
make style
make quality
make test
make typecheck
```

Local docs preview with mike:

```bash
make serve-docs
```

The API reference section is generated from source code docstrings.
