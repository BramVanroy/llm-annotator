# Provider setup

This page summarizes which extra to install, which client to use, and how to
configure authentication for each provider.

## Provider matrix

| Provider | Extra to install | Client class | Default auth source |
| --- | --- | --- | --- |
| vLLM (offline) | `llm-annotator[vllm]` or `llm-annotator[vllm-flashinfer]` | `VLLMOfflineClient` | No API key. Runs local model weights. |
| vLLM server (OpenAI-compatible) | `llm-annotator[vllm]` | `VLLMClient` | No API key by default (`api_key="EMPTY"`). |
| OpenAI | `llm-annotator[openai]` | `OpenAIClient` | `OPENAI_API_KEY` |
| Anthropic Claude | `llm-annotator[anthropic]` | `ClaudeClient` | `ANTHROPIC_API_KEY` |

## Install extras

```bash
uv add "llm-annotator[vllm]"
uv add "llm-annotator[vllm-flashinfer]"  # Faster if your hardware supports it
uv add "llm-annotator[openai]"
uv add "llm-annotator[anthropic]"
```

## Environment variables

Set only the variables you need for the provider you use:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

For Hugging Face Hub uploads from annotation jobs, authenticate with one of:

```bash
export HF_TOKEN="..."
```

(For Windows users using PowerShell, use the `$env:MYVAR = "myvalue"` syntax.)

## Examples by provider

### OpenAI

```python
from llm_annotator import Annotator, OpenAIClient

client = OpenAIClient(model="gpt-4o-mini")
with Annotator(client=client) as anno:
    ...
```

You can also pass credentials directly:

```python
client = OpenAIClient(
    model="gpt-4o-mini",
    api_key="...",
    base_url="https://api.openai.com/v1",
)
```

### Anthropic Claude

```python
from llm_annotator import Annotator, ClaudeClient

client = ClaudeClient(model="claude-sonnet-4-20250514")
with Annotator(client=client) as anno:
    ...
```

### vLLM server

```python
from llm_annotator import Annotator, VLLMClient

client = VLLMClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:8000/v1",
)
with Annotator(client=client) as anno:
    ...
```

### vLLM offline

```python
from llm_annotator import Annotator, VLLMOfflineClient

client = VLLMOfflineClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=4096,
)
with Annotator(client=client) as anno:
    ...
```
