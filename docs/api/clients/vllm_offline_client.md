# vLLM Offline Client

::: llm_annotator.clients.vllm_offline_client

## Adaptive batching on out-of-memory errors

When running inference on large datasets the batch passed to
[`batch_generate`][llm_annotator.clients.vllm_offline_client.VLLMOfflineClient.batch_generate]
can exceed available GPU memory. `batch_generate` is decorated with
[`auto_reduce_batch_size`][llm_annotator.clients.vllm_offline_client.auto_reduce_batch_size],
which automatically splits the message list into chunks and halves the chunk
size whenever a CUDA out-of-memory error is detected, retrying the failing
chunk at the smaller size.

The chunk size is controlled by the `batch_size` and `min_batch_size`
constructor arguments:

```python
from llm_annotator import VLLMOfflineClient, VLLMRuntimeOptions

messages = [
    [{"role": "user", "content": text}]
    for text in my_texts  # (1)!
]

opts = VLLMRuntimeOptions(max_tokens=256, temperature=0.0)

with VLLMOfflineClient(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=4096,
    batch_size=64,     # (2)!
    min_batch_size=1,  # (3)!
) as client:
    responses = client.batch_generate(messages=messages, options=opts)
```

1. Build one conversation per input text.
2. Start by processing 64 conversations per vLLM call. On OOM this halves to
   32, then 16, and so on.
3. Re-raise the OOM error if the chunk size would drop below this value.
   Defaults to `1`.

When `batch_size` is `None` (the default) all messages are sent in a single
call, mirroring the original behaviour while still recovering automatically
if that single call triggers an OOM.

OOM detection walks the full exception chain, so it works whether the raw
`torch.cuda.OutOfMemoryError` propagates directly or is wrapped inside a
[`ProviderError`][llm_annotator.clients.exceptions.ProviderError] (the default
behaviour with `on_error="raise"`).

!!! note
    `batch_size` controls only the number of conversations sent in a single
    Python call to vLLM. It is independent of the `max_num_seqs` constructor
    argument, which governs the vLLM scheduler and requires reloading the
    model to change.
