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

`use_batch_api=True` submits a batch to the OpenAI Batch API instead of
sending one request per sample over a thread pool. It has a completion window
of up to 24 hours and costs less, at the price of latency.
`batch_poll_interval` sets how many seconds pass between two status polls of a
running job. Both are constructor settings, so a config enables them under
`init`:

```yaml
client:
  provider: openai
  model: gpt-4o-mini
  init:
    use_batch_api: true
    batch_poll_interval: 30
```

`timeout` and `max_retries` are constructor settings too, on `OpenAIClient` and
on `ClaudeClient`. Both default to the value their SDK uses (600 seconds, two
retries).

#### Migration

`use_batch_api` and `poll_interval` are gone from `batch_generate`. Pass
`use_batch_api` and `batch_poll_interval` to the constructor instead:

```python
# before
client.batch_generate(messages=messages, use_batch_api=True, poll_interval=30)
# after
client = OpenAIClient(
    model="gpt-4o-mini", use_batch_api=True, batch_poll_interval=30
)
client.batch_generate(messages=messages)
```

Every client's `batch_generate` now takes the same three arguments
(`messages`, `options`, `gen_kwargs`). `VLLMOnlineClient` does not accept
`use_batch_api` at all, so the `ConfigurationError` it raised for
`use_batch_api=True` is gone with it.

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

A batch is one `/v1/chat/completions` request per sample, and the whole batch
is sent at once. vLLM schedules the requests it holds as one continuous batch,
so the GPU sees the same workload that a single combined request would give
it, while the result stays per sample: each response carries its own token
count, and a sample that fails (a prompt over the context length, say) is the
only one that errors.

Three constructor arguments size that:

- `timeout` (3600 seconds): how long one request may take, the wait in the
  server's queue included. A batch of 256 long generations on a busy server
  can take many minutes; a timeout below that turns a healthy run into errors.
- `max_retries` (2): retries by the OpenAI SDK, which covers connection
  errors, request timeouts and the status codes 408, 409, 429 and 5xx.
- `max_workers` (`None`): how many requests of a batch go out at once.
  `None` sends all of them. Lower it only for a server shared with other
  jobs.

Every request in flight holds one thread of the client process, which waits for
the server and uses no CPU meanwhile. A pool has `servers` x
`max_concurrent_batches_per_client` x `batch_size` of them: 4 servers with the
defaults (4 batches of 256) is 4096 threads. If the client job hits a thread or
process limit (`ulimit -u`), lower `batch_size` or set `max_workers`.

From a config file they are `init` keys:

```yaml
client:
  provider: vllm_online
  model: meta-llama/Llama-3.2-3B-Instruct
  init:
    timeout: 7200
    max_retries: 2
```

#### Migration

The client no longer posts to vLLM's own `/v1/chat/completions/batch` route.
Nothing in a config or in the Python API names that route, so no setting
changes, but two behaviours do:

- `{prefix}num_tokens` is filled on a `vllm_online` step. It used to be
  `None`, because the batch route reported one `usage` block for the whole
  batch. Code that treated the column as always empty on this provider (a
  filter, a throughput report) now gets real numbers.
- A failing sample no longer takes its batch down with it. A batch in which
  one prompt is too long used to write errors for every row of that batch.

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

#### Sizing GPU throughput

`VLLMOfflineClient.batch_generate` hands every conversation it receives to one
`vllm.LLM.chat` call; there is no client-side batch size. vLLM profiles the model at
start-up and reserves the KV cache from `gpu_memory_utilization` (vLLM's own default is
0.92; this client's default is 0.90) before any request runs. Once requests are queued,
the scheduler decides how many run in one iteration, bounded by `max_num_seqs` (maximum
sequences per iteration) and `max_num_batched_tokens` (maximum tokens per iteration).
`max_model_len` is the longest sequence (prompt plus output) that the engine accepts. In
a config file these settings live under the step's `engine` block.

The defaults are a good start, because the scheduler fills the reserved memory by
itself. What is left to set:

- The annotator's `batch_size` only decides how many samples go to one `LLM.chat` call,
  and therefore how often results reach the progress files. It does not decide how much
  work runs on the GPU at once, but a call cannot run more sequences than it holds, so
  keep `batch_size` at or above `max_num_seqs` (256 by default in this client).
- A value of several times `max_num_seqs` keeps the GPU fuller near the end of a call.
  `LLM.chat` adds every conversation of the call as a request before it steps the engine
  and returns only once all of them finished
  (`vllm/entrypoints/offline_utils.py`, `_render_and_add_requests` and `_run_engine`), and
  the scheduler admits a waiting request as soon as a running one finishes, up to
  `max_num_seqs` (`vllm/v1/core/sched/scheduler.py`). With `batch_size == max_num_seqs`
  nothing is left to admit once the first sequences finish, so the last iterations of
  every call run on the few sequences with the longest outputs. With a larger
  `batch_size` the waiting queue refills those slots. Two costs: a crash loses the rows
  of the call that was in flight, and the progress files (and the Hub backup) are written
  once per call rather than more often. `sort_by_length` does not help here, since it
  sorts on prompt length and the barrier is set by output length.
- vLLM refuses to start when the KV cache cannot hold one sequence of `max_model_len`
  tokens. Lower `max_model_len` to what your prompts and outputs need.
- An out-of-memory error means that too little memory is left next to the KV cache.
  Lower `gpu_memory_utilization`.

#### Migration

`VLLMOfflineClient(batch_size=..., min_batch_size=...)` and `init: {batch_size: ...}` in
a config are gone, together with the decorator `auto_reduce_batch_size` that halved a
chunk on a CUDA out-of-memory error. Remove those arguments and set the annotator's own
`batch_size` instead. A config with `min_batch_size` under a `vllm_offline` step's `init`
fails at load time with "Unknown 'init' keys for provider 'vllm_offline'", and one with
`batch_size` there with "'init' sets 'batch_size'. A run has one batch size, the client
block's own 'batch_size', which decides how many samples go to the provider per call."
