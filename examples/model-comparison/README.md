# LLM-as-a-judge: which model can we use for synthetic data generation?

Before spending a serving budget on generating training data at scale (as
`wiki-nl-mcq` and `finemath-dutch` do), it helps to know which candidate model
is actually reliable enough for the job. This example answers that with an
LLM-as-a-judge comparison instead of guessing from benchmark leaderboards.

By default we have three models generate synthetic question/answer pairs
grounded in 500 FineWiki articles. These are then judged by a larger dense
model.

The three generators were chosen as they are the best in their base model class
according to our prior analysis (speed/EuroEval performance). The expectation is
of course that the larger MoE model does better, but at least we can quantify by
how much in this experiment.

Qwen 3.8 27B (deterministic, no-thinking) is chosen as judge since it is the most
recent and best-in-class model available.

## Models

Optimal VLLM parameters are chosen after running datatrove's speed benchmark.

| Role | Model | `client` block |
| --- | --- | --- |
| Generator | `ibm-granite/granite-4.1-3b-fp8` | `batch_size: 16`, served with `--max-num-seqs 1024 --max-num-batched-tokens 16384` |
| Generator | `ibm-granite/granite-4.1-8b-fp8` | `batch_size: 16`, served with `--max-num-seqs 256 --max-num-batched-tokens 16384` |
| Generator | `google/gemma-4-26B-A4B-it` | `batch_size: 16`, served with `--max-num-seqs 256 --max-num-batched-tokens 8192`, MTP speculative decoding (4 tokens, draft `google/gemma-4-26B-A4B-it-assistant`) |
| Judge | `Qwen/Qwen3.8-27B-FP8` | `temperature: 0.0, repetition_penalty: 1.0`, thinking off, served with `--reasoning-parser qwen3 --speculative-config {"method":"qwen3_next_mtp","num_speculative_tokens":2}` |

## How it works

Five stages, run in this order:

1. `prepare_seed.py` filters `HuggingFaceFW/finewiki` (`nl` config) for
   non-stub articles short enough to fit one context window, and samples 500 of
   them. All three generators see exactly the same articles.
2. each `generate/pipeline-*.yaml` is one single-step pipeline per generator. Each
   reads the same seed and writes its own `question`/`answer` pair per article
   to its own `output_dir`. Three separate configs to make parallel running
   on slurm easier and because they do not depend on each other so no need to
   pass ALL the data around between them
3. `combine.py` reads the three `final` datasets, reports each model's
   generation-stage reliability (schema-valid rate and error rate, i.e. whether
   the model could produce well-formed JSON at all), keeps only the valid rows,
   and stacks them into one long dataset of 1,500 rows tagged with
   `source_model`.
4. `judge/pipeline.yaml` is one judging pass over all 1,500 rows.
   `source_model` rides along as an ordinary column the judge prompt never
   references, so it cannot bias the rubric.
5. `compare_models.py` groups the judged rows by `source_model` and prints (and
   optionally writes as CSV) the leaderboard.

## Rubric

`judge/schemas/judge.json`, asked once per generated pair. Every field is an
integer 1-5, with the anchors for 1, 3 and 5 spelled out per criterion in
`judge/prompts/judge_qa.md`:

| Field | 1 | 5 |
| --- | --- | --- |
| `question_relevant` | Off-topic, or asks something trivial | Targets a concrete, substantive fact the article really covers |
| `question_self_contained` | Meaningless without the article ("in dit artikel…") | Fully understandable to a reader who never saw the article |
| `answer_correct` | Contradicts the article | Entirely correct according to the article |
| `answer_grounded` | Contains fabricated or contradicting claims | Every claim explicitly supported, no outside knowledge |
| `fluency` | Stilted or machine-like, many errors | Reads as written by a native speaker, error-free |
| `coherence` | Does not answer the question | Answers it precisely and completely |

### Ranking

`compare_models.py`'s reliability score is

```
reliability = generation_valid_rate * (mean(answer_correct, answer_grounded) - 1) / 4
```

which is the trust-critical half of the rubric, rescaled from 1-5 onto 0-1 and
scaled by how often the model produced well-formed JSON at all. Fluency and
coherence are deliberately excluded from the ranking and reported separately: a
model that writes fluent, well-formed nonsense should still rank below one that
is a little rougher but grounded, because fluency says nothing about whether the
data is trustworthy. `overall` (the mean over all six) is reported next to it as
the all-round quality number.

## Running it

We run step-by-step. Data processing can be done within the current process
but GPU-required generation is submitted to slurm. The three generate
pipelines are independent of each other, so each `submit_pipeline.sh` call
below can be submitted in parallel; `combine.py` and the judge pipeline both
need all three to have finished first.

```sh
# 1. Seed: 500 clean articles, shared by all three generators.
uv run examples/model-comparison/prepare_seed.py \
  --num-samples 500 --num-proc 16 \
  --out examples/model-comparison/outputs/seed

# 2. Generate
# Each pipeline carries its own serving profile in its `engine:` block, so the
# submissions differ only by config.
ANNOTATE_CONFIG=examples/model-comparison/generate/pipeline-granite-4.1-3b.yaml \
  ./slurm/submit_pipeline.sh

ANNOTATE_CONFIG=examples/model-comparison/generate/pipeline-granite-4.1-8b.yaml \
  ./slurm/submit_pipeline.sh

ANNOTATE_CONFIG=examples/model-comparison/generate/pipeline-gemma-4-26b-a4b.yaml \
  ./slurm/submit_pipeline.sh

# 3. Combine + report generation-stage reliability. Run once all three jobs
# above have finished.
uv run --frozen examples/model-comparison/combine.py \
  --out examples/model-comparison/outputs/combined-qa

# 4. Judge all 1,500 pairs in one pass. Thinking is disabled on the client-side
# by changing the chat template in the config file
ANNOTATE_CONFIG=examples/model-comparison/judge/pipeline.yaml \
  ./slurm/submit_pipeline.sh

# 5. The leaderboard (bootstrap only, so no dependency beyond the project's).
uv run --frozen examples/model-comparison/compare_models.py \
  --judged examples/model-comparison/outputs/judge/final \
  --generation-stats examples/model-comparison/outputs/combined-qa_generation_stats.json \
  --out examples/model-comparison/outputs/leaderboard.csv
```

Stages 3-5 are plain CPU scripts, so nothing here submits the whole chain as
one slurm dependency graph; run each stage once the one before it has
finished, or wrap the commands above in your own submission script if you
want that automated.

Smoke-test first: run `prepare_seed.py --num-samples 20` and one generate
pipeline against a single local vLLM server before submitting the full
comparison.

## Why three pipelines, not one

The natural design is one pipeline: three `generate-*` steps (one per generator)
followed by three `judge-*` steps, chained, all reading and writing columns of
one growing dataset. That is exactly the shape `examples/pipeline-qa` uses when
every step shares one model.

It does not work here, for two independent reasons:

- Schema columns are not step-prefixed. Per the note in `docs/pipeline.md`, an
  `output_schema`'s top-level properties become plain dataset columns. Three
  `generate-*` steps writing the same `{question, answer}` schema back-to-back
  would collide on the second step (the property already exists from the first)
  unless every step's schema used a `rename` map. That is solvable, but it buys
  nothing here, since nothing downstream needs the three models' outputs as
  side-by-side columns of one row; `combine.py` wants them stacked as separate
  rows instead.
- The four stages want four different sampling profiles as well as four
  different models. That part is no longer a blocker -- `engine:` is per step,
  and each server job reads its own with `llm-annotate --serve-args` -- but
  folding the stages back together would mean every generator step also carrying
  a `rename` map for a column nothing downstream reads side by side.

Splitting into one single-model pipeline per stage sidesteps the collision
entirely: each model's `question`/`answer` lives in its own dataset until
`combine.py` stacks them by row instead of by column.

The stages could be collapsed into fewer submissions now that serving is per
step, but the `rename` maps that needs would still leave `combine.py`'s
row-stacking as the simplest way to get from "three columns per article" to "one
row per (article, model)" for the judge and the leaderboard.

## Output

Everything below is relative to
`examples/model-comparison/outputs/`.

- `generate/<model>/final`: one row per seed article, per model: `title`,
  `text`, `url`, `question`, `answer`, plus the annotator's own
  `generate-qa_valid_fields` / `generate-qa_error` / `generate-qa_finish_reason`
  / `generate-qa_num_tokens`.
- `combined-qa`: the three above, valid rows only, stacked with `source_model`.
- `combined-qa_generation_stats.json`: per-model `rows_total`, `rows_valid`,
  `valid_rate`, `error_rate`, `finish_reasons`, `avg_num_tokens`.
- `judge/final`: the combined rows, plus every rubric field from
  `judge/schemas/judge.json`.
- `leaderboard.csv`: one row per model: `<field>_mean` / `_ci_low` / `_ci_high`
  for all six rubric fields and for `overall`, the `trust_mean` behind the
  ranking, the generation-stage rates, and the reliability score with its own
  interval.
- `leaderboard_pairwise.csv`: one row per (model pair, field): `n_paired`,
  `mean_diff`, `ci_low`, `ci_high` and `wins_a`. Written next to `--out` by
  default; override with `--pairs-out`.

## Deliberately not done here

- No filtering into training data. This example measures models, it does not
  produce a training mixture. Once a model wins, generate its full-scale
  corpus with a pipeline shaped like `examples/wiki-nl-mcq` instead.
- No inter-rater reliability estimate. One judge scores every pair once, so
  nothing here measures how stable that judge's own 1-5 ratings are. The
  confidence intervals therefore describe sampling variation over articles, not
  rating noise: re-judging the same pairs would move the means somewhat. Judging
  each pair `k` times (or with `k` judges) and reporting agreement would quantify
  that, at `k` times the GPU cost.
- No significance testing. The report stops at bootstrap intervals, on purpose.
  A pairwise interval that straddles zero is not evidence the two models are
  equivalent; read its width to see what differences remain plausible. The seed
  size was chosen for cost, not to detect a specific effect size.
