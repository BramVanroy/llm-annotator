"""Build the per-model reliability leaderboard from the judged QA pairs.

Reads ``judge/pipeline.yaml``'s final dataset (every row already carries the
judge's rubric plus the ``source_model`` that survived ``combine.py``) and
groups by ``source_model``. Combined with the generation-stage stats from
``combine.py``, this is the full answer to "how reliable is this model for
synthetic data generation": how often it produces a well-formed pair at all,
and, when it does, how often the judge calls that pair relevant, grounded,
correct, un-hallucinated, and fluent Dutch.

```sh
uv run examples/gpt-nl-e/model-comparison/compare_models.py \
    --judged outputs/gpt-nl-e/model-comparison/judge/final \
    --generation-stats outputs/gpt-nl-e/model-comparison/combined-qa_generation_stats.json \
    --out outputs/gpt-nl-e/model-comparison/leaderboard.csv
```
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from typing import Any


BOOL_FIELDS = (
    "question_relevant",
    "question_self_contained",
    "answer_correct",
    "answer_grounded",
)
SCORE_FIELDS = ("fluency", "coherence", "grammar", "overall_quality")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate judged synthetic QA pairs into a per-model"
            " reliability leaderboard."
        )
    )
    parser.add_argument(
        "--judged",
        required=True,
        help="The judge pipeline's <output_dir>/final directory.",
    )
    parser.add_argument(
        "--generation-stats",
        default=None,
        help="Optional *_generation_stats.json written by combine.py.",
    )
    parser.add_argument(
        "--out", default=None, help="CSV file to write the leaderboard to."
    )
    return parser


def reliability(row: dict[str, Any]) -> float:
    """Score a model's overall reliability for synthetic data generation.

    A model that writes fluent nonsense should rank below one that is a
    little rougher but grounded, so this multiplies the rates that matter
    most for training-data trustworthiness rather than averaging every
    field the judge reports.

    Args:
        row: One model's aggregated leaderboard row.

    Returns:
        A score in ``[0, 1]``; higher is more reliable.
    """
    gen_rate = row.get("generation_valid_rate")
    gen_rate = gen_rate if gen_rate is not None else 1.0
    return float(
        gen_rate
        * row["answer_correct_rate"]
        * row["answer_grounded_rate"]
        * (1 - row["hallucinated_rate"])
    )


def main(args: list[str] | None = None) -> None:
    """Aggregate the judged dataset into a per-model leaderboard and print it.

    Args:
        args: Optional command-line arguments.
    """
    from datasets import Dataset

    parsed = build_parser().parse_args(args)

    ds = Dataset.load_from_disk(parsed.judged)
    print(f"Loaded {len(ds):,} judged pairs from {parsed.judged}")

    generation_stats: dict[str, dict[str, Any]] = {}
    if parsed.generation_stats:
        with open(parsed.generation_stats, encoding="utf-8") as fh:
            generation_stats = json.load(fh)

    per_model: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "hallucinated": 0,
            **dict.fromkeys(BOOL_FIELDS, 0),
            **{field: [] for field in SCORE_FIELDS},
        }
    )

    for sample in ds:
        agg = per_model[sample.get("source_model") or "unknown"]
        agg["n"] += 1
        if sample.get("hallucinated"):
            agg["hallucinated"] += 1
        for field in BOOL_FIELDS:
            if sample.get(field):
                agg[field] += 1
        for field in SCORE_FIELDS:
            value = sample.get(field)
            if value is not None:
                agg[field].append(value)

    rows: list[dict[str, Any]] = []
    for model_name, agg in per_model.items():
        n = agg["n"] or 1
        row: dict[str, Any] = {
            "source_model": model_name,
            "n_judged": agg["n"],
        }
        for field in BOOL_FIELDS:
            row[f"{field}_rate"] = round(agg[field] / n, 4)
        row["hallucinated_rate"] = round(agg["hallucinated"] / n, 4)
        for field in SCORE_FIELDS:
            scores = agg[field]
            row[f"{field}_mean"] = (
                round(sum(scores) / len(scores), 3) if scores else None
            )
        gen = generation_stats.get(model_name)
        if gen:
            row["generation_valid_rate"] = gen.get("valid_rate")
            row["generation_error_rate"] = gen.get("error_rate")
        rows.append(row)

    rows.sort(key=reliability, reverse=True)

    print("\nModel reliability leaderboard (most to least reliable):")
    for row in rows:
        print(
            f"\n{row['source_model']}"
            f"  (n={row['n_judged']}, reliability={reliability(row):.3f})"
        )
        if "generation_valid_rate" in row:
            print(
                f"  generation:  schema-valid"
                f" {row['generation_valid_rate']:.1%}"
                f"  errors {row['generation_error_rate']:.1%}"
            )
        print(
            "  judged:      relevant {question_relevant_rate:.1%}"
            "  self-contained {question_self_contained_rate:.1%}"
            "  correct {answer_correct_rate:.1%}"
            "  grounded {answer_grounded_rate:.1%}"
            "  hallucinated {hallucinated_rate:.1%}".format(**row)
        )
        print(
            "  language:    fluency {fluency_mean}"
            "  coherence {coherence_mean}"
            "  grammar {grammar_mean}"
            "  overall {overall_quality_mean}".format(**row)
        )

    if parsed.out and rows:
        fieldnames = list(rows[0].keys())
        with open(parsed.out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {parsed.out}")


if __name__ == "__main__":
    main()
