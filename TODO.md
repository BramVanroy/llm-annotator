# TODO

Feature ideas, ordered by how much they help a large run. Each item states the current
behaviour (checked against v0.16.0), what goes wrong, a proposal, and the benefit. Line
references point at `src/llm_annotator/`.

## 1. Let a run grow: raise `max_num_samples` and reuse every finished row

A common workflow is a pilot on a few thousand samples, then the full run, then a larger run
months later. Today that works for a pipeline with one step and corrupts a pipeline with more
than one step. The corruption raises no error and the final row count looks right.

### What already works

`annotate_dataset` adds the `idx` column as the source row number before it shuffles with
`shuffle_seed` and selects `max_num_samples` (`annotator.py`, around line 406). With the same
seed and an unchanged source, the first N rows of the permutation are a prefix of the first M
rows for every M > N. `_get_skip_idxs` reads the finished `idx` values from the progress
backups. So a one-step pipeline that is rerun with a larger cap sends only the new rows to the
model.

### What goes wrong

- A finished step is skipped while `<NN>-<step>/output/state.json` exists (`_is_complete` in
  `pipeline.py`), and that check never looks at `max_num_samples`. A user who raises the cap
  and reruns gets the old result back, with no message that the cap was ignored. The only way
  forward is to delete `output/` and `final/` by hand.
- `prepare_data` returns a leftover `annotate/<prefix>prepared_dataset` without a look at
  `max_num_samples` (`annotator.py`, around line 909). A finished step deletes that cache, a
  crashed step leaves it behind, and it then pins the old selection.
- `--overwrite` looks like the natural way to "redo with more samples", and it deletes the
  progress backups, which is the one thing that an extension needs.
- The output of a step is sorted by `idx` and the column is then dropped (`_post_annotate`,
  around lines 1707 and 1730). The next step numbers its rows again by position. After an
  extension the new rows sit between the old ones, so every old row has a new position, while
  the progress backups of step 2 still hold the old positions. Step 2 then skips rows that
  were never processed and processes again rows that were.

  Measured with a 40 row dataset, a two-step pipeline and a stub server: cap 10, then cap 20
  with `output/` and `final/` deleted. Step 1 sent exactly the 10 new rows. Step 2 sent 10
  rows of which 5 had been judged before, and 5 rows were never judged. The final dataset had
  20 rows: 5 duplicates and 5 missing. Texts and verdicts stay together (`keep_columns=True`
  writes the whole input row), so no verdict lands on the wrong text. The damage is missing
  and doubled rows.

### Proposal

- Carry one stable sample id through every step. Keep the `idx` of the first step as the
  identity of a row in all later steps (do not drop it between steps, and do not number rows
  again by position). A step that expands one row into several can derive child ids from the
  parent id.
- Record `max_num_samples`, `shuffle_seed` and a fingerprint of the source dataset next to
  `state.json` and next to the prepared dataset. When the config asks for more samples than
  the record holds, treat the step as unfinished and resume it. When the seed or the source
  fingerprint changed, stop with an error that says the finished rows cannot be reused.
- Add a documented way to extend (for example `llm-annotate config.yaml --extend`), which
  keeps the progress backups, rebuilds the prepared dataset and reruns every step on the new
  rows only. Make `--overwrite` state in its help text that it deletes finished work.
- Add a page "Growing a run" to the docs that says what must stay unchanged (the source
  dataset and `shuffle_seed`).

### Benefit

A pilot is never wasted: its rows are the first rows of the full run. A dataset can grow when
budget allows, at the cost of the new rows only. Without this, a user of a multi-step pipeline
has two choices: pay again for every finished generation, or split the data into shards by
hand and run each shard in its own output directory.

## 2. Set the sample cap from the command line

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

Pilot, full run and extension become one config file and one number on the command line. A
job script can take the number from an environment variable. Today each size needs an edited
copy of the YAML, and the copy also has to fix relative prompt and schema paths through
`config_dir`.

## 3. Start the client on a partial server pool under SLURM

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

## 4. Progress files on a large run

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

## 5. Reject a `queue_size` that is too small when the config loads

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
