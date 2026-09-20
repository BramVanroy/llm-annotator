# TODO

Feature ideas, ordered by how much they help a large run. Each item states the current
behaviour (checked against v0.16.0), what goes wrong, a proposal, and the benefit. Line
references point at `src/llm_annotator/`.

## 1. Set the sample cap from the command line

### Current behaviour

`dataset.max_num_samples` and `dataset.shuffle_seed` can only be set in the YAML. The CLI
overrides `output_dir`, `hub_id`, `log_level` and `overwrite` (`pipeline.py`, around line 620),
and the config loader has no environment interpolation. `load_pipeline_config(overrides=...)`
can already replace a top-level key, so the value is reachable from Python and hidden from
the CLI.

### Proposal

`--max-num-samples N` and `--shuffle-seed S`, or a general `--set dataset.max_num_samples=N`
that works for every key. The resolved value is already written to `pipeline.json`, so a run
stays reproducible.

### Benefit

A pilot, the full run and a later extension already share one config file, since a run can now
grow by raising `max_num_samples` and re-running. This proposal removes the last edit: the
number reaches the config from the command line instead of the YAML, so a job script can take
it from an environment variable. Today changing the cap means editing the YAML directly, and a
one-off size that should not touch the tracked config still needs a copy of the file, which
also has to fix its relative prompt and schema paths through `config_dir`.

## 2. Start the client on a partial server pool under SLURM

### Current behaviour

`pool.min_servers` exists (`PoolConfig`, `config.py` around line 410), and the client can
start once that many servers answer. `describe_steps()` reports `servers` and
`gpus_per_vllm_server` and leaves `min_servers` out. The SLURM scripts read the pool from
`describe_steps()`, so their wait loop can only wait for the full pool.

### Proposal

Add `min_servers` to the output of `describe_steps()`, and let the wait loop in
`slurm/vllm_annotate.sh` start the client when `min_servers` servers are ready. The background
watcher that 0.15 added already admits servers that arrive later.

### Benefit

On a busy GPU partition the last server of an array of eight can sit in the queue for hours.
Today the seven servers that did start hold their GPUs idle for that time, and a client that
waits longer than its pool timeout fails the whole step.

## 3. Progress files on a large run

### Current behaviour

`max_samples_per_output_file` defaults to 1000. A step over 500,000 rows leaves 500 JSONL
files, and `_get_skip_idxs` parses every one of them at each resume. On a shared network
filesystem that costs minutes per restart, and a run that is preempted often pays it often.

### Proposal

Scale the default with the size of the dataset (for example 1% of the rows, with a floor of
1000), or merge the progress files into one when a resume starts. Say in the docs that the
value trades the size of the loss at a crash against the cost of a resume.

### Benefit

Faster restarts on preemptible partitions and fewer small files on filesystems that have a
file count quota.

## 4. Reject a `queue_size` that is too small when the config loads

### Current behaviour

`_resolve_queue_size` raises any `queue_size` below `clients x max_concurrent_batches_per_client`
to that product and logs a warning. With 8 servers and the default of 4 batches in flight per
client, `queue_size: 8` becomes 32. The warning is one line in a long job log, and the YAML
keeps a value that has no effect.

### Proposal

Validate the value when the config is loaded, with an error that states the minimum. Or drop
the key and always derive it. `--describe-steps` can print the effective value, together with
the total number of requests in flight (`servers x max_concurrent_batches_per_client x
batch_size`), which is the number that a user needs for sizing `max_num_seqs` on the server.

### Benefit

No dead settings in configs, and the real concurrency of a pool is visible before any GPU job
is submitted. `max_concurrent_batches_per_client` arrived in 0.15 with a default of 4, so a
config written for 0.14 now sends four times the requests per server that its author sized it
for, and nothing reports that.
