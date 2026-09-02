# Persona-grounded Dutch Wikipedia QA, with reasoning traces

Synthetic instruction data for Dutch: a persona asks a question about a
Wikipedia article, a reasoning model answers it from that article, a second
model scores the pair, and what survives becomes an SFT set in two flavours,
with and without the reasoning trace.

The grounding article is what keeps the answers factual, but it is deliberately
not part of the training sample. The question is written to stand on its own and
the answer to read as if it came from the model's own knowledge, so a model
trained on this answers Dutch questions directly rather than learning to quote a
document it will not be given.

## Models

| Role | Model | `client` block |
| --- | --- | --- |
| Question + answer | `Qwen/Qwen3.6-35B-A3B-FP8` | `batch_size: 16`, 4 servers; thinking off for the question, on for the answer, which is served with `--reasoning-parser qwen3` |
| Judge | `Qwen/Qwen3.6-27B-FP8` | `temperature: 0.0`, `repetition_penalty: 1.0`, thinking off, `--reasoning-parser qwen3` as a safety net |

The MoE generator does the bulk work at 3B active parameters per token; the
dense model judges, so the thing being scored and the thing scoring it are not
the same weights.

## How it works

Five stages, run in this order:

1. `prepare_seed.py` filters `HuggingFaceFW/finewiki` (`nl` config) for
   non-stub articles, splits anything longer than one context window into
   several chunks along its `## ` section headings instead of dropping it,
   drops articles and chunks that look malformed or non-prose (see
   "Chunking and quality gates" below), samples N chunks, and gives each one
   a random persona from `nvidia/Nemotron-Personas-Belgium` (`nl_BE` split),
   a random question type, and a random question/answer length.
2. `generate/pipeline-qa.yaml` is one config with two steps on one model:
   `write-question` writes free-text questions, renamed straight to a
   `question` column, then `answer-question` answers it with thinking on. Its
   trace lands in `answer-question_reasoning` and the answer alone in
   `answer-question_response`, because that step is served with
   `--reasoning-parser qwen3`.
3. `filter_rows.py` drops the rows that are unusable before any GPU is spent
   judging them: provider errors, answers cut off at the token budget, empty
   answers, and answers that refer to the article they were shown.
4. `judge/pipeline.yaml` scores every surviving pair on seven criteria.
5. `build_sft.py` applies the rubric thresholds and writes the `plain` and
   `reasoning` configs.

Stages 1, 3 and 5 are plain CPU scripts; 2 and 4 want GPUs and go to Slurm.

## Chunking and quality gates

An article longer than `--max-words` is split on its `## ` section headings
rather than dropped: `pack_sections` glues consecutive whole sections into
chunks first-fit, in order, so every chunk is at most `--max-words` words and
none overlap. A section that alone exceeds `--max-words` (rare, but "History"
sections can run long) is cut to the last complete sentence that fits, and
whatever is left of it is discarded rather than carried into another chunk.
Sentence splitting uses a blank spaCy Dutch pipeline with only a rule-based
sentencizer (the `spacy` extra: `uv sync --extra spacy`), no model download
needed.

Two gates run alongside that, both there because finewiki's markdown is not
always well-formed:

- `keep_section_sizes` drops an article outright if one of its sections runs
  into the hundreds of thousands of characters. That is not a long section,
  it is leftover markup or a raw table dump, and past a point it would blow
  spaCy's own `nlp.max_length` (1,000,000 characters by default) besides.
  `--max-section-chars` (default 200,000) is set far above any real section
  so this only catches that kind of malformed content.
- `keep_running_text` drops a chunk that reads as a table or bullet list
  rather than running prose. Neither carries normal sentence-ending
  punctuation, so the sentencizer reads a table as one giant "sentence"
  (average words/sentence far above `--max-avg-sentence-words`, default 60)
  or a list of short items as a run of short fragments (average far below
  `--min-avg-sentence-words`, default 4). Ordinary prose lands in between.

Both gates are heuristics tuned against `HuggingFaceFW/finewiki`'s actual
mess, not a general prose classifier; read a sample of what they drop with
`--num-samples 500` before trusting the defaults on a very different corpus.

## Where the variability comes from

Half a million questions written from one prompt collapse into one voice asking
one kind of question, at one length. Four cheap levers spread them out, all
decided in `prepare_seed.py` rather than by the model:

- **Persona.** A one-sentence Belgian-Dutch persona per article ("Ilse Sebrechts
  is een uiterst gedisciplineerde verzekeringsconsulent uit Oostende...") shapes
  which detail the question picks up and how formally it is phrased. Personas
  are drawn with replacement, so the pool of 300k covers any run size.
- **Question type.** One of six Dutch instructions (feit, definitie, uitleg,
  chronologie, vergelijking, gevolg), assigned at random. Letting the model pick
  its own type instead collapses onto "feit" for most articles, which is exactly
  the diversity problem the persona is there to solve.
- **Question length.** Either a one-sentence question, or a few sentences that
  open with an invented, generic reason for asking (out of interest, for work,
  for a project, because a kid asked) before the question itself. 60/40 short
  to long.
- **Answer length.** Either one paragraph or two to four, weighted per row by
  `question_type`: `uitleg`, `chronologie`, `vergelijking` and `gevolg` draw
  the longer option 70% of the time, `feit` and `definitie` only 15%, so the
  answer's depth tracks what the question actually needs instead of varying
  independently of it.

The persona reaches `write-question` and stops there. It is who *asks*; an
answer conditioned on it would put the persona into the assistant turn, which is
the half that becomes the training target. The prompt also forbids naming the
persona's literal profession, name, city or age inside the question, so the
persona shows up as register and angle rather than as a stated fact; only a
long question may add a generic reason for asking ("voor mijn werk", "mijn
kleinkind vroeg me dit"), and even then it must stay generic rather than quote
the persona description. For the same reason the judge never sees the persona:
a judge that knew it would reward questions that describe their asker.

## Rubric

`judge/schemas/judge.json`, asked once per pair. Every field is an integer 1-5,
with the anchors for 1, 3 and 5 spelled out per criterion in
`judge/prompts/judge_qa.md`:

| Field | 1 | 5 |
| --- | --- | --- |
| `question_answerable` | The article says nothing about it | The article contains everything a full answer needs |
| `question_self_contained` | Meaningless without the article | Fully understandable to a reader who never saw it |
| `question_natural` | An exam question about a text | A question someone would really ask |
| `answer_correct` | Contradicts the article | Entirely correct according to the article |
| `answer_grounded` | Contains fabricated claims | Every claim explicitly supported |
| `answer_complete` | Does not answer the question | Answers it fully, without padding |
| `fluency` | Stilted or machine-like | Reads as written by a native speaker |

`build_sft.py` keeps a pair when every criterion is at least `--min-score`
(default 4) and the two trust-critical ones, `answer_correct` and
`answer_grounded`, are at least `--min-grounding` (default 5). A wrong answer is
worse training data than a stiff one, so those two are held to a higher bar than
fluency.

Source-reference leakage ("volgens dit artikel") is caught by a phrase list in
`filter_rows.py` rather than by the judge: the phrasing is fixed enough for a
regex, and catching it there costs nothing and spends no GPU on rows that would
be thrown away anyway.

## Reasoning traces

`answer-question` is the step whose trace is worth keeping, so it runs with
`enable_thinking: true` and is served with `--reasoning-parser qwen3`. Without
the parser vLLM leaves `<think>...</think>` inside the message content; with it,
the trace arrives separately and the annotator writes it to
`answer-question_reasoning` (added in v0.13.0). Neither step has an
`output_schema`: both the question and the answer are free text.

A trace that talks about "het artikel" is a problem the answer prompt cannot
fully prevent: at inference time there is no article, so such a trace teaches
the model to cite a source it does not have. `filter_rows.py` flags those as
`reasoning_mentions_source` instead of dropping them, and `build_sft.py` leaves
them out of the `reasoning` config while keeping their answer in `plain`. Pass
`--keep-source-mentions` to keep them.

`engine` is replaced wholesale by a step override rather than merged key by key,
so `answer-question` repeats the whole block instead of just the parser line.
Check any change to it with `llm-annotate <config> --serve-args answer-question`.

## Running it

```sh
# 0. Once per clone: prepare_seed.py needs spaCy for sentence-level
# truncation and the running-text quality gate.
uv sync --dev --extra spacy

# 1. Seed. Defaults to every chunk that passes the quality filter.
# Optional: relevant for certain SLURM envs: point the datasets cache at the node's
# local disk, not the NFS-mounted home: num-proc workers writing shards to a
# network filesystem concurrently is what turns this into a multi-hour crawl
# instead of a few minutes.
export HF_HOME=/tmp/$USER/hf_home
uv run --frozen examples/wiki-nl-persona-qa/prepare_seed.py \
  --num-proc "$(nproc)" \
  --out examples/wiki-nl-persona-qa/outputs/seed

# 2. Question + answer, two chained steps on one model.
# You may have to execute this command multiple commands to complete
# it on your hardware but do not worry: rerunning the script just continues
# where it left of
CLIENT_TIME=04:00:00 SERVER_TIME=03:30:00 ANNOTATE_CONFIG=examples/wiki-nl-persona-qa/generate/pipeline-qa.yaml \
  ./slurm/submit_pipeline.sh

# 3. Filter, once the generate jobs have finished.
uv run --frozen examples/wiki-nl-persona-qa/filter_rows.py \
  --out examples/wiki-nl-persona-qa/outputs/qa-split

# 4. Judge.
ANNOTATE_CONFIG=examples/wiki-nl-persona-qa/judge/pipeline.yaml \
  ./slurm/submit_pipeline.sh

# 5. The two SFT configs.
uv run --frozen examples/wiki-nl-persona-qa/build_sft.py \
  --out examples/wiki-nl-persona-qa/outputs/sft
```

Nothing here submits the whole chain as one dependency graph: stages 3 and 5 run
in the shell between submissions. Smoke-test first with
`prepare_seed.py --num-samples 20` and `OUTPUT_DIR=.../smoke`, and read the rows
before spending the real allocation. A prompt bug and a truncation bug both look
like "it ran fine".

`pool.servers`, `max_num_seqs` and `max_num_batched_tokens` are the throughput
knobs. The values in the configs are placeholders that schedule and run; take
the real ones from a speed benchmark for these models on your GPUs.

## Output

Relative to `examples/wiki-nl-persona-qa/outputs/`:

- `seed`: `title`, `text`, `url`, `chunk_index`, `num_chunks`, `persona`,
  `question_type`, `question_length`, `answer_length`. `chunk_index` /
  `num_chunks` are `0`/`1` for an article that fit within `--max-words`
  untouched, and mark which piece of a split article `text` holds otherwise.
- `generate/final`: the seed plus `question`, `answer-question_response` (the
  answer), `answer-question_reasoning` (the trace) and both steps' bookkeeping
  columns.
- `qa-split`: `idx`, `title`, `url`, `text`, `persona`, `question_type`,
  `question_length`, `answer_length`, `question`, `reasoning`, `answer`,
  `has_reasoning`, `reasoning_mentions_source`, `num_tokens`.
- `qa-split_filter_stats.json`: rows in and out, the keep rate, how many rows
  carry a trace, and a count per drop reason.
- `judge/final`: the above plus the seven rubric fields.
- `sft/plain`: `messages` (user question, assistant answer), the rubric scores,
  `quality_score`, and `idx` / `title` / `url` / `persona` / `question_type` /
  `question_length` / `answer_length` for provenance.
- `sft/reasoning`: the subset that has a usable trace. Same columns, with the
  assistant turn rendered as `<think>\n...\n</think>\n\nanswer`, plus `thinking`
  and `content` as separate columns so the trace can be re-rendered for a
  template that wants it elsewhere.
- `sft/sft_stats.json`: rows judged, rows kept per config, the thresholds used,
  the rubric means of the kept rows, and the question-type distribution.

## Deliberately not done here

- **No article in the training rows.** The article is grounding for generation,
  not context for the trained model. A retrieval-style variant that keeps it as
  a system message is a different dataset; build it by carrying `text` through
  `build_sft.py`.
- **No deduplication.** Chunks are sampled without replacement, but a long
  article split into several chunks can end up contributing more than one of
  them, and two personas can still produce near-identical questions about the
  same popular topic. Near-duplicate filtering on the question column belongs
  before training, not here.
- **One judge, one pass.** Nothing measures how stable the judge's own 1-5
  ratings are, so the thresholds are a quality filter, not a calibrated
  measurement. Judging a sample twice would quantify that, at twice the cost.
- **No model comparison.** `Qwen3.6-35B-A3B-FP8` is assumed fit for the job. To
  establish that rather than assume it, run `examples/model-comparison/` over a
  few hundred articles first.
