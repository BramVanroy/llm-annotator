# Choosing a provider

Four providers, and the hardware you have picks one. The config name in the
heading is what `provider:` takes in a [config file](pipeline.md); the same
choice from Python is which client class you construct.

## One GPU: `vllm_offline`

`VLLMOfflineClient` loads the model into the process that runs the annotation
and calls `vllm.LLM.chat`. No server, no ports, no second job. It is the
simplest thing that works and the fastest for a single GPU, because the
annotator and the engine share one process.

```yaml
client:
  provider: vllm_offline
  model: Qwen/Qwen3-8B
  engine:
    max_model_len: 8192
```

A model that needs more than one GPU still fits here: set
`engine.tensor_parallel_size` to the number of GPUs on the node. There is no
client-side batch size, and the annotator's `batch_size` only decides how often
results reach the progress files. What to set instead, and why, is in
[Sizing GPU throughput](provider-info.md#sizing-gpu-throughput).

## Several GPUs or several nodes: a `vllm_online` pool

One process cannot drive two model replicas, so more than one GPU worth of
throughput means more than one vLLM server. Point `vllm_online` at several of
them and the run uses a
[`VLLMQueueAnnotator`][llm_annotator.annotator.VLLMQueueAnnotator], which hands
each batch to whichever server is free and keeps the single set of JSONL
progress files.

```yaml
client:
  provider: vllm_online
  model: Qwen/Qwen3-8B
  engine:
    tensor_parallel_size: 2   # GPUs per server
  pool:
    servers: 4                # four such servers
    min_servers: 2            # start once two answer /health
```

`servers` and `engine` are read by a job submitter, not by the library.
[`slurm/`](slurm.md) is one: it submits a GPU server array plus a CPU client per
step, waits for `min_servers` and releases the GPUs when the step ends. A pool
started some other way is handed to the run with `base_urls`, `hosts_file` or
`url_glob`, see [Many vLLM servers](pipeline.md#many-vllm-servers).

The count of requests in flight is `servers` times
`max_concurrent_batches_per_client` times `batch_size`, and each one holds a
thread of the client process. The three constructor arguments that size that
(`timeout`, `max_retries`, `max_workers`) are described under
[vLLM online](provider-info.md#vllm-online-server).

A single server is the same provider without the pool: one `base_urls` entry,
or none at all when the client may ask the server what it serves. Use it when
something else already started the server, or when the annotation runs on a
machine that has no GPU of its own.

## No GPU: `openai` or `claude`

`OpenAIClient` and `ClaudeClient` send one request per sample over a thread
pool, so the run needs nothing but CPU and a network route.

```yaml
client:
  provider: claude
  model: claude-haiku-4-5
  options:
    max_completion_tokens: 512
```

Both read their key from the environment (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`), see [Provider setup](provider-info.md). `OpenAIClient`
also points at any OpenAI-compatible endpoint through `base_url`, and
`init.use_batch_api: true` submits through the OpenAI Batch API instead, which
costs less and can take up to 24 hours.

These are the providers where a wasted request costs money, so the next section
matters most here.

## What a retry costs

`num_retries_invalid` sends an invalid sample to the model again, up to that
many times, and it defaults to 5. Every attempt is a paid request: 300 samples
that stay invalid cost 1800 requests rather than 300. Set it to 0 to write the
first answer whatever it is.

A sample is invalid when either of two flags says so.

`{prefix}valid_fields` is `False` when:

- the request errored, so there is no response to read;
- the response does not parse as JSON;
- it parses to something other than an object (`[1, 2]`, `"text"`, `null`, `3`);
- it leaves out a property that the schema's `required` list names.

A schema with no `required` list therefore counts as valid as soon as the
response parses to an object, however few properties it holds. Add `required`
to the schema when a missing property should be retried. A property the schema
declares and the response leaves out is `None` on that row; a key the response
adds and the schema does not declare is dropped, with one warning per run.

`{prefix}valid` is what your own `validate_fn` returned, and only exists when
you passed one (Python API only).

Two ways to spend less:

- Make the schema strict enough that a good answer passes on the first try, and
  give `max_completion_tokens` enough room for the whole object. A truncated
  JSON object is the usual cause of an invalid sample.
- Develop the prompt against `vllm_offline` or a small pilot
  (`--max-num-samples 50`) before pointing the config at a paid endpoint. The
  same config runs on either provider; only the `client` block changes.

A sample that is still invalid after the retries is written as it is, with the
flags set, so nothing is lost. [Troubleshooting](troubleshooting.md) says how to
read those columns and how to redo the rows later.
