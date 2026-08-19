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
# you can pre-install most pre-compiled kernels so they
# do not need to be JIT-compiled. Especially useful in
# SLURM/server settings
uv add "llm-annotator[vllm,vllm-kernels]"
uv add "llm-annotator[openai]"
uv add "llm-annotator[anthropic]"
```

`vllm-kernels` pulls prebuilt FlashInfer wheels that are not on PyPI; this
repo's `pyproject.toml` points at FlashInfer's own index only for its own
`uv sync`, and that routing is not published with the package. If you are
adding `llm-annotator[vllm,vllm-kernels]` from PyPI into another project,
you need to point your own installer at those indexes too:

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

```bash
# pip: install the kernels explicitly
pip install "llm-annotator[vllm]" \
    flashinfer-cubin==0.6.16.post3 flashinfer-jit-cache==0.6.16.post3 \
    --extra-index-url https://flashinfer.ai/whl/ \
    --extra-index-url https://flashinfer.ai/whl/cu130/
```

Without one of the above, `vllm-kernels` will fail to resolve and vLLM
falls back to JIT-compiling FlashInfer at startup (needs `nvcc`).

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
