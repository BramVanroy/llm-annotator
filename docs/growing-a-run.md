# Growing a run

A run can start as a pilot on a few thousand samples and grow later. Raise
`dataset.max_num_samples` and run the same command again on the same `output_dir`. No extra flag
is needed. Every step of the pipeline resumes, and rows that are already finished are never sent
to the model again.

The cap can be raised in the config file or on the command line with `--max-num-samples`. The two
are the same setting; the flag exists so that one tracked config can serve the pilot, the full run
and a job script that takes the number from an environment variable.

## The workflow

Start with a pilot:

```yaml title="pilot.yaml"
output_dir: outputs/imdb-sentiment

dataset:
  name: stanfordnlp/imdb
  split: test
  max_num_samples: 2000
  shuffle_seed: 42

client:
  provider: vllm_offline
  model: Qwen/Qwen3-8B

steps:
  - name: sentiment
    prompt: "Classify the sentiment: {text}"
  - name: confidence
    prompt: "How confident are you in this label: {sentiment_response}? Text: {text}"
```

Run it, look at the result, then raise the cap and run the identical command again:

```yaml title="pilot.yaml"
dataset:
  name: stanfordnlp/imdb
  split: test
  max_num_samples: 50000
  shuffle_seed: 42
```

```bash
llm-annotate pilot.yaml
```

Or leave the file alone and pass the cap in:

```bash
llm-annotate pilot.yaml --max-num-samples 2000     # the pilot
llm-annotate pilot.yaml --max-num-samples 50000    # the full run
```

Both steps resume. The 2000 rows of the pilot are not sent to the model again. The 48000 new rows
go through `sentiment` and then through `confidence`.

## What must stay the same

Two things must match the finished run:

- `shuffle_seed`, unset or set to the same value.
- The source dataset, unless it only had rows appended to it (see below).

`max_num_samples` is the value that changes, and it can only go up.

## What is rejected

!!! warning "Nothing is deleted before the check"

    Something that would give the finished rows another meaning raises a `ValueError` that names
    the cause, before any file is touched:

    - A changed `shuffle_seed`. With another seed the permutation is different, so the rows behind
      a given id are no longer the ones that were judged.
    - A changed source dataset, other than rows appended to a source that has no `shuffle_seed`.
    - A lower `max_num_samples` than the finished run selected. Shrinking the selection would drop
      rows that already have an answer.

The error message names three ways out: restore the old value, use a new `output_dir` for the
different run, or pass `--overwrite` (`overwrite: true`) to discard the finished work and start
over. `--overwrite` deletes the directories of the selected steps, including every finished
generation in them; it is not needed for a higher cap, since a plain re-run already annotates only
the new rows.

## Appending rows without a shuffle

One kind of source change is accepted: rows appended to a dataset that has no `shuffle_seed`. The
old rows keep the row numbers they had before, so their ids, and the answers already recorded under
those ids, still match. Editing or removing a row that was already annotated is rejected, because
the row at that position no longer matches what the finished run saw there.

The same rule is what lets a `generate` step grow: its source is the `prompts` list or file.
Appending new prompts and running the same config again annotates only the new entries. Editing or
removing a prompt that was already generated is rejected for the same reason as any other source
change.

## Running one step at a time, and under SLURM

Under `--steps`, each job is handed only its own step, and each one notices on its own whether that
step's finished selection matches the config. Resubmitting the same pipeline with a higher cap
therefore works one job at a time, the same way
[running one step at a time](pipeline.md#running-one-step-at-a-time) already works for a plain
resume. Selecting only a later step while an earlier one is still out of date raises an error that
says to run the earlier step first, or to select it too.

The SLURM submitter takes the same `--set KEY=VALUE` as the CLI and forwards it to every step job
of one submission:

```sh
./slurm/submit_pipeline.sh --set dataset.max_num_samples=50000 pilot.yaml
```

`dataset.shuffle_seed` and any other config key work the same way. Set the cap on the submission
rather than per job, so that all steps of one run agree about how many rows they select.

## Runs made by an older version

A run whose first step finished under llm-annotator 0.16 or older has no selection record. Its
later steps identified a row by its position in their input, and that position changes when the
input grows. Such a run cannot grow. `run_pipeline` compares `dataset.max_num_samples` and
`dataset.shuffle_seed` in the config against the values recorded in `<output_dir>/pipeline.json`
from the first run, and raises when they differ and the first step has no selection record. An
unchanged config keeps working as before.

## Limits

- A source loaded by Hub id (`dataset.name`) is compared only when the prepared data is rebuilt; a
  plain re-run that reuses the local or Hub cache does not download the source to check it.
- The source signature probes 64 rows. An edit to a row outside that probe is not detected.
- A prepared-data backup restored from a Hub branch, on a machine with no local selection record,
  is reused as is; there is nothing to compare it against.
- `sort_by_length`, `batch_size` and the client settings may change between runs freely.
- A cap given with `--max-num-samples` is compared exactly like one written in the config: the run
  is judged on the resolved value, which `<output_dir>/pipeline.json` records.
- A changed prompt or a changed `output_schema` is not detected. Finished rows keep the answers
  they already have.

## From Python

The same growth works without a config file. Call
[`Annotator.annotate_dataset`][llm_annotator.annotator.Annotator.annotate_dataset] again with a
higher `max_num_samples`, on the same `output_dir`:

```python
ds = anno.annotate_dataset(
    output_dir="outputs/imdb-sentiment",
    prompt_template="Classify the sentiment: {text}",
    dataset_name="stanfordnlp/imdb",
    dataset_split="test",
    max_num_samples=50_000,
    shuffle_seed=42,
)
```

Chaining several `annotate_dataset` calls the way a pipeline chains steps needs two more keyword
arguments: `keep_idx_column=True` on every call, so the id is not dropped from the result, and
`reuse_idx_column=True` on every call after the first, so the next call treats the incoming
`idx_column` as the sample id instead of raising on a column that already exists.

## Why it works

The first step assigns the `idx_column` (`idx` by default) as the row number in the source
dataset, before it shuffles with `shuffle_seed` and cuts at `max_num_samples`. With a fixed seed and
an unchanged source, the shuffle is deterministic: the first N rows of the permutation are a prefix
of the first M rows for every M greater than N. Raising the cap therefore keeps every row the pilot
already saw, in the same order, and adds new rows after them. Every later step keeps that `idx`
instead of numbering its rows again by position.

[`Annotator.prepare_data`][llm_annotator.annotator.Annotator.prepare_data] writes a
[`SelectionRecord`][llm_annotator.annotator.SelectionRecord] next to the prepared data, at
`<output_dir>/<task_prefix>selection.json` (inside a pipeline:
`<NN>-<name>/annotate/<name>_selection.json`). It holds `max_num_samples`, `shuffle_seed`, a
signature of the source dataset, the source row count and the number of rows the cap selected. A
later run compares its request against that record through
[`SelectionRecord.is_stale`][llm_annotator.annotator.SelectionRecord.is_stale]: a higher cap with
the same seed and source rebuilds the prepared data and resumes the step, and rows already in the
step's progress files are not sent to the model again.

The source signature comes from
[`dataset_signature`][llm_annotator.utils.dataset_signature]: a SHA256 hash of the row count, the
column names and up to 64 evenly spaced rows. A changed row count always changes it; an edit to a
row that the signature does not probe does not.
