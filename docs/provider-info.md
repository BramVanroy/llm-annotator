# Provider setup

This page summarizes which extra to install, which client to use, and how to
configure authentication for each provider.

## Provider matrix

| Provider | Config name | Extra to install | Client class | Default auth source |
| --- | --- | --- | --- | --- |
| vLLM offline (in-process) | `vllm_offline` | `llm-annotator[vllm]` | `VLLMOfflineClient` | No API key. Runs local model weights. |
| vLLM online (OpenAI-compatible server) | `vllm_online` | `llm-annotator[openai]` | `VLLMOnlineClient` | No API key by default (`api_key="EMPTY"`). |
| OpenAI | `openai` | `llm-annotator[openai]` | `OpenAIClient` | `OPENAI_API_KEY` |
| Anthropic Claude | `claude` | `llm-annotator[anthropic]` | `ClaudeClient` | `ANTHROPIC_API_KEY` |

The **config name** column is the exact spelling `provider:` takes in a
[config file](pipeline.md); no other spelling is accepted. Note that the online
vLLM client speaks the OpenAI protocol, so it needs the `openai` extra rather
than the (much heavier) `vllm` one — that extra is only needed where the model
weights are actually loaded.

## Install extras

```bash
uv add "llm-annotator[vllm]"
uv add "llm-annotator[openai]"
uv add "llm-annotator[anthropic]"
```

## Prebuilt vLLM kernels

vLLM runs attention through FlashInfer. Without prebuilt kernels it
JIT-compiles them on the first request, which needs `nvcc` on the node and
races between servers that share `~/.cache/flashinfer`. Pre-installing them is
worth it wherever you serve models, and close to mandatory on a cluster.

There is no `vllm-kernels` extra to install them for you.
`flashinfer-jit-cache` is not on PyPI (it is published per CUDA version on
FlashInfer's own index) and the `flashinfer-cubin` on PyPI trails the releases
vLLM pins against, so an extra would only resolve for people who had already
added those indexes to their own project. Install the two wheels by hand
instead, pinned to the `flashinfer-python` version vLLM pulled in and to the
CUDA version your torch wheel was built against:

```bash
version=$(python -c "import importlib.metadata as m; print(m.version('flashinfer-python'))")
cuda=cu$(python -c "import torch; print(torch.version.cuda.replace('.', ''))")

uv pip install "flashinfer-cubin==$version" --index-url https://flashinfer.ai/whl/
uv pip install "flashinfer-jit-cache==$version" --index-url "https://flashinfer.ai/whl/$cuda/"
```

To keep the pins in your own project rather than installing imperatively,
route the two packages to those indexes explicitly:

```toml
# uv: copy these into your project's pyproject.toml
[[tool.uv.index]]
name = "flashinfer-cubin"
url = "https://flashinfer.ai/whl/"
explicit = true

[[tool.uv.index]]
name = "flashinfer-jit-cache"
url = "https://flashinfer.ai/whl/cu130/"
explicit = true

[tool.uv.sources]
flashinfer-cubin = { index = "flashinfer-cubin" }
flashinfer-jit-cache = { index = "flashinfer-jit-cache" }
```

This repo does exactly that for its own checkout, where the kernels live in
the `vllm-kernels` dependency group:

```bash
uv sync --extra vllm --group vllm-kernels
```

A dependency group is not published in the wheel metadata, so that routing
only ever has to hold here.

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

### vLLM online (server)

```python
from llm_annotator import Annotator, VLLMOnlineClient

client = VLLMOnlineClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    base_url="http://localhost:8000/v1",
)
with Annotator(client=client) as anno:
    ...
```

### vLLM offline (in-process)

```python
from llm_annotator import Annotator, VLLMOfflineClient

client = VLLMOfflineClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=4096,
)
with Annotator(client=client) as anno:
    ...
```
