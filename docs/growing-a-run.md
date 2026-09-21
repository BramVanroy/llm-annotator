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

Every setting that decides what a row's prompt and answer mean must match the finished run, or
the run is rejected (see below):

- `shuffle_seed`, unset or set to the same value.
- The source dataset, unless it only had rows appended to it (see below).
- The prompt template.
- The system message.
- `sort_by_length`.
- `idx_column`.
- `reuse_idx_column`.
- `preprocess_fn`.
- The output schema.

`max_num_samples` is the value that changes, and it can only go up.

## What is rejected

!!! warning "Nothing is deleted before the check"

    Something that would give the finished rows another meaning raises a `ValueError` that names
    every cause, before any file is touched:

    - A changed `shuffle_seed`. With another seed the permutation is different, so the rows behind
      a given id are no longer the ones that were judged.
    - A changed source dataset, other than rows appended to a source that has no `shuffle_seed`.
    - A lower `max_num_samples` than the finished run selected. Shrinking the selection would drop
      rows that already have an answer.
    - An edited prompt template, system message, `sort_by_length`, `idx_column`,
      `reuse_idx_column` or `preprocess_fn`. The rows that are already annotated answer a question
      the new request does not ask.
    - An edited output schema. The columns a finished row has depend on it, so mixing two schemas
      in one output is refused the same way.

The message names every setting that changed, for example:

```text
The finished rows in 'outputs/run/progress_backup' cannot be reused: the prompt template changed.
Restore the old value(s), use a new 'output_dir', or overwrite the run ('overwrite=True') to
discard the finished rows and annotate every sample again. A run can only grow through a higher
'max_num_samples' with the same settings, or through rows appended to a source that is not
shuffled.
```

A pipeline names the command instead of `overwrite=True`, see
[Editing a step](pipeline.md#editing-a-step).

Three ways out: restore the old value, use a new `output_dir` for the different run, or pass
`--overwrite` (`overwrite: true`) to discard the finished work and start over. `--overwrite`
deletes the directories of the selected steps, including every finished generation in them; it is
not needed for a higher cap, since a plain re-run already annotates only the new rows.

## Editing a prompt mid-run

Prompt development usually means a short run, a look at the answers, an edit, and another run. Stop
the pilot from above partway through, change the prompt of the `sentiment` step and run the
identical command again:

```yaml title="pilot.yaml"
steps:
  - name: sentiment
    prompt: "Classify the sentiment as positive, negative or neutral: {text}"   # edited
```

```bash
llm-annotate pilot.yaml
```

The rows that `sentiment` already finished hold answers to the old prompt, so the run stops with the
`ValueError` above. Pick one of the three ways out: put the old prompt back and keep those answers,
point `output_dir` somewhere else to keep both versions on disk, or discard the old answers with
`--overwrite`. In a pipeline the error names the command itself, here
`llm-annotate pilot.yaml --steps sentiment confidence --overwrite`: the edited step and the steps
that read it are annotated again, and the steps before it keep their results. See
[Editing a step](pipeline.md#editing-a-step).

An edit made before any row has been annotated has nothing to conflict with: the prepared data is
rebuilt from the new prompt and the run continues, with an INFO line that names what changed.

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

## Migrating an output directory

An `output_dir` from a release whose `selection.json` recorded only the sample selection cannot be
resumed by this version. Such a record says nothing about the prompt, so a resume could not tell
whether the finished rows answer the prompt that is asked for now. Reading it raises:

```text
The record in 'outputs/imdb-sentiment/selection.json' does not describe the settings that its run
was annotated with, so a resume cannot tell whether the prompt still matches. Finish the run with
the release that wrote it, annotate it into a new 'output_dir', or overwrite it. See 'Migrating an
output directory' in docs/growing-a-run.md.
```

Three ways forward:

- Finish the run with the release that wrote it, and let the next run start under this one.
- Point `output_dir` at a new directory, which annotates every row again and leaves the old result
  on disk.
- Pass `--overwrite` (`overwrite: true`, or `overwrite=True` from Python) to discard the finished
  rows in place and annotate every row again.

A fourth way keeps the finished rows, at your own risk: delete the `selection.json` files under
`output_dir` (inside a pipeline, `<NN>-<name>/annotate/<name>_selection.json`). Only the rows that
never ran are then sent to the model. Nothing is compared, and the settings of that run are
recorded as the ones the whole directory was annotated with. If the prompt changed since the
finished rows were written, the output holds answers to both prompts and no later run can tell.
Use it only when you know the settings did not change.

```bash
rm outputs/imdb-sentiment/*selection.json
llm-annotate pilot.yaml
```

## Limits

- A source loaded by Hub id (`dataset.name`) is compared only when the prepared data is rebuilt; a
  plain re-run that reuses the local or Hub cache does not download the source to check it.
- The source signature probes 64 rows. An edit to a row outside that probe is not detected.
- Prepared data reused with no record at all, which is a backup restored from the Hub branch on a
  machine that never ran the preparation, is taken at face value: a WARNING says it is reused as it
  is, and the current run's settings are recorded then, so a later edit to them is caught. The
  record itself is local only; it is not stored on the Hub.
- `batch_size` and the client settings may change between runs freely; they do not affect what the
  prepared data holds.
- A cap given with `--max-num-samples` is compared exactly like one written in the config: the run
  is judged on the resolved value, which `<output_dir>/pipeline.json` records.
- `preprocess_fn` is compared by its qualified name plus a hash of its source. When the source
  cannot be read (a `functools.partial`, a C callable, a function defined in a REPL), a warning
  says so and an edit to it is not detected.

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
`<NN>-<name>/annotate/<name>_selection.json`). It holds `max_num_samples`, the source row count,
the number of rows the cap selected, and `components`: one short string per setting listed under
[What must stay the same](#what-must-stay-the-same), keyed by the setting's name (a hash for a
long value such as the prompt template, the value itself for a short one such as `shuffle_seed`). A
later run compares its request against that record through
[`SelectionRecord.changed_components`][llm_annotator.annotator.SelectionRecord.changed_components],
which names the settings that differ: a higher cap with the same seed and source rebuilds the
prepared data and resumes the step, and rows already in the step's progress files are not sent to
the model again.

The source signature comes from
[`dataset_signature`][llm_annotator.utils.dataset_signature]: a SHA256 hash of the row count, the
column names and up to 64 evenly spaced rows. A changed row count always changes it; an edit to a
row that the signature does not probe does not.
