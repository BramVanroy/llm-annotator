"""Build the per-model reliability leaderboard from the judged QA pairs.

Reads ``judge/pipeline.yaml``'s final dataset where each row has a synthetic
QA pair as well as the ``source_model`` that generated it.

Combined with the generation-stage stats from ``combine.py``, this is the
full answer to "how reliable is this model for synthetic data generation":
how often it produces a well-formed pair at all, and, when it does,
how the judge rates that pair on each rubric criterion.

Every rubric is a 1-5 score, so every number in the report is a mean, and
every mean comes with a 95% bootstrap percentile interval, following Koehn
(2004).

```sh
uv run examples/model-comparison/compare_models.py \
    --judged examples/model-comparison/outputs/judge/final \
    --generation-stats examples/model-comparison/outputs/combined-qa_generation_stats.json \
    --out examples/model-comparison/outputs/leaderboard.csv
```
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from typing import Any


RUBRIC_FIELDS = (
    "question_relevant",
    "question_self_contained",
    "answer_correct",
    "answer_grounded",
    "fluency",
    "coherence",
)

# The criteria that decide whether the data can be trusted, as opposed to
# whether it reads well. These drive the ranking; see ``reliability``.
TRUST_FIELDS = ("answer_correct", "answer_grounded")

# The mean over the rubric fields and the mean over the trust fields
OVERALL_FIELD = "overall"
TRUST_FIELD = "trust"

COMPARE_FIELDS = (OVERALL_FIELD, *RUBRIC_FIELDS)
BOOTSTRAP_FIELDS = (*COMPARE_FIELDS, TRUST_FIELD)

SCORE_MIN = 1
SCORE_MAX = 5
CONFIDENCE = 0.95


def bootstrap_means(matrix: Any, n_resamples: int, seed: int) -> Any:
    """Resample rows with replacement and average each resample.

    This is Koehn's bootstrap: draw ``n`` observations with replacement from
    the ``n`` we have, take the mean, repeat.

    Args:
        matrix: ``(n_samples, n_fields)`` array of observations.
        n_resamples: Number of bootstrap resamples to draw.
        seed: Seed for the resampling.

    Returns:
        An ``(n_resamples, n_fields)`` array of resampled means.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    n_samples = matrix.shape[0]
    draws = rng.integers(0, n_samples, size=(n_resamples, n_samples))
    return matrix[draws].mean(axis=1)


def percentile_ci(distribution: Any) -> Any:
    """Take the percentile interval of a bootstrap distribution.

    Args:
        distribution: ``(n_resamples,)`` or ``(n_resamples, n_fields)``
            array of resampled statistics.

    Returns:
        ``(low, high)`` arrays (or scalars) cutting off an equal tail on
        either side, together covering ``CONFIDENCE``.
    """
    import numpy as np

    tail = (1.0 - CONFIDENCE) / 2.0
    return np.quantile(distribution, [tail, 1.0 - tail], axis=0)


def normalise(score: float) -> float:
    """Map a 1-5 rubric score onto ``[0, 1]``.

    Args:
        score: A score on the judge's 1-5 scale.

    Returns:
        The same score rescaled so that 1 maps to 0 and 5 maps to 1.
    """
    return (score - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


def collect(
    ds: Any, pair_column: str
) -> tuple[dict[str, dict[Any, Any]], int]:
    """Group the judged rows by model and seed article.

    Args:
        ds: The judged dataset.
        pair_column: Column identifying the seed article.

    Returns:
        ``(per_model, num_dropped)``, where ``per_model`` maps a model name
        to ``{article_id: {field: score}}`` and ``num_dropped`` counts rows
        the judge left without a complete rubric.
    """
    per_model: dict[str, dict[Any, Any]] = {}
    dropped = 0

    for sample in ds:
        scores = {field: sample.get(field) for field in RUBRIC_FIELDS}
        if any(value is None for value in scores.values()):
            dropped += 1
            continue
        scores = {field: float(value) for field, value in scores.items()}
        scores[OVERALL_FIELD] = sum(
            scores[field] for field in RUBRIC_FIELDS
        ) / len(RUBRIC_FIELDS)
        scores[TRUST_FIELD] = sum(
            scores[field] for field in TRUST_FIELDS
        ) / len(TRUST_FIELDS)

        model = sample.get("source_model") or "unknown"
        article = sample.get(pair_column)
        per_model.setdefault(model, {}).setdefault(article, scores)

    return per_model, dropped


def _matrix(samples: list[dict[str, Any]]) -> Any:
    """Stack per-sample score dicts into a ``BOOTSTRAP_FIELDS``-wide array.

    Args:
        samples: Per-sample score dicts, as produced by ``collect``.

    Returns:
        An ``(n_samples, len(BOOTSTRAP_FIELDS))`` float array.
    """
    import numpy as np

    return np.array(
        [[sample[field] for field in BOOTSTRAP_FIELDS] for sample in samples],
        dtype=float,
    )


def summarise(
    per_model: dict[str, dict[Any, Any]],
    generation_stats: dict[str, dict[str, Any]],
    n_resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build one leaderboard row per model.

    Args:
        per_model: Output of ``collect``.
        generation_stats: ``combine.py``'s per-model generation stats.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for the bootstrap resampling.

    Returns:
        The leaderboard rows, sorted from most to least reliable.
    """
    rows: list[dict[str, Any]] = []
    for model, articles in per_model.items():
        samples = list(articles.values())
        matrix = _matrix(samples)
        means = matrix.mean(axis=0)
        resampled = bootstrap_means(matrix, n_resamples, seed)
        low, high = percentile_ci(resampled)

        row: dict[str, Any] = {
            "source_model": model,
            "n_judged": len(samples),
        }
        for index, field in enumerate(BOOTSTRAP_FIELDS):
            row[f"{field}_mean"] = round(float(means[index]), 3)
            if field in COMPARE_FIELDS:
                row[f"{field}_ci_low"] = round(float(low[index]), 3)
                row[f"{field}_ci_high"] = round(float(high[index]), 3)

        # Always set both keys, even without stats for this model: the CSV
        # writer takes its header from the first row, so the rows must not
        # differ in shape.
        gen = generation_stats.get(model) or {}
        row["generation_valid_rate"] = gen.get("valid_rate")
        row["generation_error_rate"] = gen.get("error_rate")
        gen_rate = float(
            1.0
            if row["generation_valid_rate"] is None
            else row["generation_valid_rate"]
        )

        # The reliability score is a fixed transform of the trust mean, so
        # pushing the bootstrap distribution of that mean through it gives
        # the score its interval for free.
        trust = BOOTSTRAP_FIELDS.index(TRUST_FIELD)
        scores = gen_rate * normalise(resampled[:, trust])
        score_low, score_high = percentile_ci(scores)
        row["reliability"] = round(
            gen_rate * normalise(float(means[trust])), 4
        )
        row["reliability_ci_low"] = round(float(score_low), 4)
        row["reliability_ci_high"] = round(float(score_high), 4)
        rows.append(row)

    rows.sort(key=lambda item: item["reliability"], reverse=True)
    return rows


def compare_pairs(
    per_model: dict[str, dict[Any, Any]], n_resamples: int, seed: int
) -> list[dict[str, Any]]:
    """Compare every pair of models per article, on every rubric field.

    Paired bootstrap: an article scored by both models is one observation,
    the observation is the difference, and articles rather than rows are what
    gets resampled. Between-article difficulty therefore cancels instead of
    being noise both models have to be measured through.

    Args:
        per_model: Output of ``collect``.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for the bootstrap resampling.

    Returns:
        One record per (model pair, field), carrying the paired mean
        difference, its confidence interval, and the share of resamples in
        which ``model_a`` came out ahead.
    """
    records: list[dict[str, Any]] = []
    for left, right in combinations(sorted(per_model), 2):
        shared = sorted(
            set(per_model[left]) & set(per_model[right]),
            key=lambda value: (value is None, value),
        )
        if not shared:
            continue
        diff = _matrix([per_model[left][key] for key in shared]) - _matrix(
            [per_model[right][key] for key in shared]
        )
        means = diff.mean(axis=0)
        resampled = bootstrap_means(diff, n_resamples, seed)
        low, high = percentile_ci(resampled)
        wins = (resampled > 0.0).mean(axis=0)

        for index, field in enumerate(BOOTSTRAP_FIELDS):
            if field not in COMPARE_FIELDS:
                continue
            records.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "field": field,
                    "n_paired": len(shared),
                    "mean_diff": round(float(means[index]), 3),
                    "ci_low": round(float(low[index]), 3),
                    "ci_high": round(float(high[index]), 3),
                    "wins_a": round(float(wins[index]), 3),
                }
            )
    return records


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` to ``path`` as CSV, using the first row's keys.

    Args:
        path: Destination file.
        rows: The records to write; a header is taken from ``rows[0]``.
    """
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def _print_leaderboard(rows: list[dict[str, Any]]) -> None:
    """Print the per-model leaderboard.

    Args:
        rows: Output of ``summarise``.
    """
    print("\nModel reliability leaderboard (most to least reliable):")
    print(
        f"All intervals are {CONFIDENCE:.0%} bootstrap percentile intervals"
        " over articles."
    )
    for row in rows:
        print(
            f"\n{row['source_model']}  (n={row['n_judged']},"
            f" reliability={row['reliability']:.3f}"
            f" [{row['reliability_ci_low']:.3f},"
            f" {row['reliability_ci_high']:.3f}])"
        )
        if row["generation_valid_rate"] is not None:
            print(
                f"  generation:  schema-valid"
                f" {row['generation_valid_rate']:.1%}"
                f"  errors {row['generation_error_rate']:.1%}"
            )
        for field in COMPARE_FIELDS:
            print(
                f"  {field:<24s} {row[f'{field}_mean']:.2f}"
                f"  [{row[f'{field}_ci_low']:.2f},"
                f" {row[f'{field}_ci_high']:.2f}]"
            )


def _print_pairs(records: list[dict[str, Any]]) -> None:
    """Print the paired model-vs-model comparisons.

    Args:
        records: Output of ``compare_pairs``.
    """
    if not records:
        print("\nNo two models share an article, so nothing to compare.")
        return

    print(
        f"\nPaired per-article differences (A - B), with a {CONFIDENCE:.0%}"
        " bootstrap interval\non the mean difference. wins = the share of"
        " resamples in which A scored\nhigher; * = the interval excludes"
        " zero."
    )
    seen: set[tuple[str, str]] = set()
    for record in records:
        pair = (record["model_a"], record["model_b"])
        if pair not in seen:
            seen.add(pair)
            print(
                f"\n{record['model_a']} - {record['model_b']}"
                f"  (n={record['n_paired']} shared articles)"
            )
        clears_zero = record["ci_low"] > 0.0 or record["ci_high"] < 0.0
        print(
            f"  {record['field']:<24s} {record['mean_diff']:+.2f}"
            f"  [{record['ci_low']:+.2f}, {record['ci_high']:+.2f}]"
            f"  wins={record['wins_a']:.0%}"
            f"{' *' if clears_zero else ''}"
        )


def compare_models(
    *,
    judged_dir: str,
    generation_stats_file: str | None,
    output_file: str | None,
    pairs_output_file: str | None,
    pair_column: str,
    n_resamples: int,
    seed: int,
) -> None:
    """Aggregate the judged dataset into a per-model leaderboard and print it.

    Args:
        judged_dir: The judge pipeline's ``<output_dir>/final`` directory.
        generation_stats_file: ``combine.py``'s stats JSON, if any.
        output_file: CSV file to write the leaderboard to, if any.
        pairs_output_file: CSV file for the paired comparisons. Defaults to
            ``<output_file>_pairwise.csv`` when a leaderboard file is given.
        pair_column: Column identifying the seed article.
        n_resamples: Number of bootstrap resamples.
        seed: Seed for the bootstrap resampling.

    Raises:
        ValueError: If no judged row carried a complete rubric.
    """
    from datasets import Dataset

    ds = Dataset.load_from_disk(judged_dir)
    print(f"Loaded {len(ds):,} judged pairs from {judged_dir}")

    generation_stats: dict[str, dict[str, Any]] = {}
    if generation_stats_file:
        with open(generation_stats_file, encoding="utf-8") as fh:
            generation_stats = json.load(fh)

    per_model, dropped = collect(ds, pair_column)
    if dropped:
        print(
            f"Skipped {dropped:,} row(s) the judge left without a complete"
            " rubric."
        )
    if not per_model:
        raise ValueError(
            "No judged row carried a complete rubric; nothing to compare."
        )

    rows = summarise(per_model, generation_stats, n_resamples, seed)
    _print_leaderboard(rows)

    records = compare_pairs(per_model, n_resamples, seed)
    _print_pairs(records)

    if output_file:
        _write_csv(output_file, rows)
    if pairs_output_file is None and output_file:
        pairs_output_file = f"{output_file.removesuffix('.csv')}_pairwise.csv"
    if pairs_output_file:
        _write_csv(pairs_output_file, records)


def main(args: list[str] | None = None) -> None:
    """Build the model reliability leaderboard from the command line.

    Args:
        args: Optional command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate judged synthetic QA pairs into a per-model"
            " reliability leaderboard with bootstrap confidence intervals"
            " and paired model-vs-model comparisons."
        )
    )
    parser.add_argument(
        "--judged",
        dest="judged_dir",
        required=True,
        help="The judge pipeline's <output_dir>/final directory.",
    )
    parser.add_argument(
        "--generation-stats",
        dest="generation_stats_file",
        default=None,
        help="Optional *_generation_stats.json written by combine.py.",
    )
    parser.add_argument(
        "--out",
        dest="output_file",
        default=None,
        help="CSV file to write the leaderboard to.",
    )
    parser.add_argument(
        "--pairs-out",
        dest="pairs_output_file",
        default=None,
        help=(
            "CSV file for the paired model-vs-model comparisons. Defaults to"
            " <--out>_pairwise.csv when --out is given."
        ),
    )
    parser.add_argument(
        "--pair-column",
        default="idx",
        help=(
            "Column that identifies the seed article, used to pair models"
            " sample-for-sample (default: idx, as written by combine.py)."
        ),
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=1000,
        help="Bootstrap resamples (default: 1000, as in Koehn 2004).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the bootstrap resampling.",
    )

    compare_models(**vars(parser.parse_args(args)))


if __name__ == "__main__":
    main()
