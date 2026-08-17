# `model-comparison` — which model can we trust for Dutch synthetic data?

Every other subproject in `gpt-nl-e` *uses* a model to generate training
data. This one answers the question those subprojects assume the answer to:
**which of our candidate models is actually reliable enough to do that?**

Three candidate models each generate a grounded question/answer pair for the
same 200 Dutch Wikipedia articles. A fourth, larger model judges every pair
against a rubric that covers both Dutch language quality (fluency, coherence,
grammar) and synthetic-data quality (is the question relevant, is the answer
correct, is it grounded in the article or hallucinated). `compare_models.py`
turns that into a per-model reliability leaderboard.

This is a decision tool, not a training-data pipeline: it produces a
leaderboard, not a dataset meant for a training mixture. Nothing here is
filtered into "keep" rows the way `to_pairs.py` scripts elsewhere in
`gpt-nl-e` are.

## Models

| Role | Model | `client` block |
| --- | --- | --- |
| Generator | `ibm-granite/granite-4.1-3b-fp8` | `batch_size: 16`, served with `--max-num-seqs 1024 --max-num-batched-tokens 16384` |
| Generator | `ibm-granite/granite-4.1-8b-fp8` | `batch_size: 16`, served with `--max-num-seqs 256 --max-num-batched-tokens 16384` |
| Generator | `google/gemma-4-26B-A4B-it` | `batch_size: 16`, served with `--max-num-seqs 256 --max-num-batched-tokens 8192`, MTP speculative decoding (4 tokens, draft `google/gemma-4-26B-A4B-it-assistant`) |
| Judge | `Qwen/Qwen3.8-27B-FP8` | `temperature: 1.0, top_p: 0.95, top_k: 20, repetition_penalty: 1.0`, served with `--reasoning-parser qwen3 --speculative-config {"method":"qwen3_next_mtp","num_speculative_tokens":2}` |

All four run at `gpu_memory_utilization=0.90`. The judge is deliberately a
different, larger model than any generator — a model cannot be trusted to
grade its own homework, which is exactly the round-trip-consistency argument
`Synthetic-data-strategies.md` §3.3 makes for judging in general.

## How it works

Four stages, run in this order:

1. **`prepare_seed.py`** — filter `HuggingFaceFW/finewiki` (`nl` config) for
   non-stub articles short enough to fit one context window, and sample 200
   of them. All three generators see exactly the same articles.
2. **`generate/pipeline-*.yaml`** — one single-step pipeline per generator.
   Each reads the same seed and writes its own `question`/`answer` pair per
   article to its own `output_dir`. Three separate configs, not three steps
   of one pipeline — see [Why three pipelines, not one](#why-three-pipelines-not-one).
3. **`combine.py`** — reads the three `final` datasets, reports each model's
   *generation-stage* reliability (schema-valid rate, error rate — whether
   the model could produce well-formed JSON at all), keeps only the valid
   rows, and stacks them into one long dataset (600 rows) tagged with
   `source_model`.
4. **`judge/pipeline.yaml`** — one judging pass over all 600 rows.
   `source_model` rides along as an ordinary column the judge prompt never
   references, so it cannot bias the rubric.
5. **`compare_models.py`** — groups the judged rows by `source_model` and
   prints (and optionally writes as CSV) the leaderboard.

## Rubric

`judge/schemas/judge.json`, asked once per generated pair:

| Field | What it measures |
| --- | --- |
| `question_relevant` | Question targets a real fact/event/definition/relation, not something trivial |
| `question_self_contained` | Question never points back at "this article" |
| `answer_correct` | Answer is factually correct according to the article |
| `answer_grounded` | Every claim in the answer is explicitly supported by the article |
| `hallucinated` | Answer states something not in, or contradicting, the article |
| `fluency` (1–5) | How natural the Dutch reads |
| `coherence` (1–5) | How well the answer addresses the question |
| `grammar` (1–5) | Grammatical correctness |
| `overall_quality` (1–5) | Overall usability as training data, weighted toward correctness/groundedness |

`compare_models.py`'s reliability score is
`generation_valid_rate × answer_correct_rate × answer_grounded_rate × (1 − hallucinated_rate)`
— a model that writes fluent, well-formed nonsense still ranks below one
that is a little rougher but grounded, because fluency alone says nothing
about whether the *data* is trustworthy.

All prompts are written in Standaardnederlands understandable in both the
Netherlands and Flanders (see `generate/prompts/` and `judge/prompts/`);
schema field names and descriptions are in English, matching every other
`gpt-nl-e` subproject.

## Running it

```sh
# 1. Seed: 200 clean articles, shared by all three generators.
uv run examples/gpt-nl-e/model-comparison/prepare_seed.py \
  --num-samples 200 \
  --out examples/gpt-nl-e/model-comparison/outputs/seed

# 2. Generate. Three separate submissions -- see the table below for the
#    env vars each one needs; they cannot be combined into one
#    submit_pipeline.sh call because serving flags are global per
#    submission (see the next section).
GPU_MEM_UTIL=0.90 \
VLLM_EXTRA_ARGS='--max-num-seqs 1024 --max-num-batched-tokens 16384' \
ANNOTATE_CONFIG=examples/gpt-nl-e/model-comparison/generate/pipeline-granite-4.1-3b.yaml \
  ./slurm/submit_pipeline.sh

GPU_MEM_UTIL=0.90 \
VLLM_EXTRA_ARGS='--max-num-seqs 256 --max-num-batched-tokens 16384' \
ANNOTATE_CONFIG=examples/gpt-nl-e/model-comparison/generate/pipeline-granite-4.1-8b.yaml \
  ./slurm/submit_pipeline.sh

GPU_MEM_UTIL=0.90 \
VLLM_EXTRA_ARGS='--max-num-seqs 256 --max-num-batched-tokens 8192 --speculative-config {"model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4}' \
ANNOTATE_CONFIG=examples/gpt-nl-e/model-comparison/generate/pipeline-gemma-4-26b-a4b.yaml \
  ./slurm/submit_pipeline.sh

# 3. Combine + report generation-stage reliability.
uv run examples/gpt-nl-e/model-comparison/combine.py \
  --out outputs/gpt-nl-e/model-comparison/combined-qa

# 4. Judge all 600 pairs in one pass.
GPU_MEM_UTIL=0.90 \
VLLM_EXTRA_ARGS='--reasoning-parser qwen3 --speculative-config {"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
ANNOTATE_CONFIG=examples/gpt-nl-e/model-comparison/judge/pipeline.yaml \
  ./slurm/submit_pipeline.sh

# 5. The leaderboard.
uv run examples/gpt-nl-e/model-comparison/compare_models.py \
  --judged outputs/gpt-nl-e/model-comparison/judge/final \
  --generation-stats outputs/gpt-nl-e/model-comparison/combined-qa_generation_stats.json \
  --out outputs/gpt-nl-e/model-comparison/leaderboard.csv
```

**Smoke-test first**, same as every other `gpt-nl-e` subproject: run
`prepare_seed.py --num-samples 20` and one generate pipeline against a
single local vLLM server before submitting the full comparison.

## Why three pipelines, not one

The natural design is one pipeline: three `generate-*` steps (one per
generator) followed by three `judge-*` steps, chained, all reading and
writing columns of one growing dataset. That is exactly the shape
`domain-queries-nl` and the other subprojects use when every step shares one
model.

It does not work here, for two independent reasons:

- **Schema columns are not step-prefixed.** Per the note in
  `../README.md`, an `output_schema`'s top-level properties become plain
  dataset columns. Three `generate-*` steps writing the same
  `{question, answer}` schema back-to-back would collide on the second step
  (the property already exists from the first) unless every step's schema
  used a `rename` map — solvable, but it buys nothing here, since nothing
  downstream needs the three models' outputs as side-by-side columns of one
  row; `combine.py` wants them stacked as separate rows instead.
- **Serving flags are global per `submit_pipeline.sh` invocation**, not
  per-step. `PoolConfig` carries `servers`/`gpus_per_vllm_server`, which
  *does* come from the config and does differ correctly per step. But
  `max-num-seqs`, `max-num-batched-tokens` and `--speculative-config` reach
  `vllm serve` only through `VLLM_EXTRA_ARGS`/`GPU_MEM_UTIL`
  (`slurm/vllm_common.sh`), which are set once on the `submit_pipeline.sh`
  command line and apply to every `vllm_pool` step's servers in that
  submission. This pipeline needs four different serving profiles (three
  generators plus the judge); one submission cannot give each step its own.
  `../README.md`'s "Serving flags that the config cannot express yet"
  section already flags this as unimplemented, and `../../README.md` names
  per-step serving arguments as a separate piece of future work.

Splitting into one single-model pipeline per stage sidesteps both problems
at once: each submission serves exactly one model, so its env vars are
unambiguous, and there is no schema collision to rename around because each
model's `question`/`answer` lives in its own dataset until `combine.py`
stacks them by row instead of by column.

If per-step serving overrides land later, the four stages could collapse
into fewer submissions; the `rename` maps needed to fold generation back
into one pipeline would still leave `combine.py`'s row-stacking as the
simplest way to get from "three columns per article" to "one row per
(article, model)" for the judge and the leaderboard.

## Output

- `outputs/gpt-nl-e/model-comparison/generate/<model>/final` — one row per
  seed article, per model: `title`, `text`, `url`, `question`, `answer`,
  plus the annotator's own `generate-qa_valid_fields` /
  `generate-qa_error` / `generate-qa_finish_reason` / `generate-qa_num_tokens`.
- `outputs/gpt-nl-e/model-comparison/combined-qa` — the three above, valid
  rows only, stacked with `source_model`.
- `outputs/gpt-nl-e/model-comparison/combined-qa_generation_stats.json` —
  per-model `rows_total`, `rows_valid`, `valid_rate`, `error_rate`,
  `finish_reasons`, `avg_num_tokens`.
- `outputs/gpt-nl-e/model-comparison/judge/final` — the 600 rows above, plus
  every rubric field from `judge/schemas/judge.json`.
- `outputs/gpt-nl-e/model-comparison/leaderboard.csv` — one row per model:
  every rubric rate/mean, the generation-stage rates, and the combined
  reliability score.

## Deliberately not done here

- **No filtering into training data.** This subproject measures models, it
  does not produce a mixture. Once a model wins, generate its full-scale
  corpus with a subproject shaped like `domain-queries-nl` and filter that
  output with a `to_pairs.py`-style script instead.
- **No per-document paired comparison.** `idx` is preserved through
  generation (it is added before `sort_by_length` reorders anything), so a
  deeper analysis *could* compare what each model wrote for the very same
  article — `compare_models.py` deliberately does not, because the judge
  scores each pair independently and a per-article diff is a different
  question from "which model is reliable overall."
