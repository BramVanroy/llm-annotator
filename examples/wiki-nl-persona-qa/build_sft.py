"""Turn the judged pairs into two SFT datasets: with and without reasoning.

Applies the rubric thresholds, then writes the same surviving pairs twice. The
`plain` config is a plain instruction-tuning set; the `reasoning` config holds
the subset that also has a usable reasoning trace, with that trace folded into
the assistant turn between `<think>` tags. Both are `messages` datasets, which
is what TRL's `SFTTrainer` and friends read directly.

```sh
uv run --frozen examples/wiki-nl-persona-qa/build_sft.py \
    --judged examples/wiki-nl-persona-qa/outputs/judge/final \
    --out examples/wiki-nl-persona-qa/outputs/sft
```
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any


RUBRIC_FIELDS = (
    "question_answerable",
    "question_self_contained",
    "question_natural",
    "answer_correct",
    "answer_grounded",
    "answer_complete",
    "fluency",
)
"""The judge rubric, in the order judge/schemas/judge.json emits it."""

GROUNDING_FIELDS = ("answer_correct", "answer_grounded")
"""The two criteria that decide whether the answer can be trusted at all, and
which therefore get their own, stricter threshold."""

CARRIED_COLUMNS = (
    "idx",
    "title",
    "url",
    "persona",
    "question_type",
)
"""Provenance kept on every SFT row. The article itself is deliberately not
carried: it is the grounding for generation, not part of the training sample,
and it would multiply the dataset size for nothing."""


def passes(row: dict[str, Any], *, min_score: int, min_grounding: int) -> bool:
    """Check one judged row against the rubric thresholds.

    Args:
        row: A judged row, carrying every field of `RUBRIC_FIELDS`.
        min_score: Lowest acceptable score on any criterion.
        min_grounding: Lowest acceptable score on the grounding criteria.

    Returns:
        Whether the row is good enough for the training set.
    """
    scores = {field: row.get(field) for field in RUBRIC_FIELDS}
    if any(score is None for score in scores.values()):
        return False
    return all(
        score >= (min_grounding if field in GROUNDING_FIELDS else min_score)
        for field, score in scores.items()
    )


def build_sft(
    *,
    judged: str,
    task_prefix: str,
    min_score: int,
    min_grounding: int,
    keep_source_mentions: bool,
    output_dir: str | None,
    hub_id: str | None,
) -> None:
    """Filter the judged pairs and write the two SFT configs.

    Args:
        judged: The judge pipeline's ``<output_dir>/final`` directory.
        task_prefix: Prefix the judge step wrote its bookkeeping columns under.
        min_score: Lowest acceptable score on any rubric criterion.
        min_grounding: Lowest acceptable score on the grounding criteria.
        keep_source_mentions: Keep reasoning traces that refer to the article
            they were shown.
        output_dir: Directory to write ``plain/`` and ``reasoning/`` under.
        hub_id: Hub dataset id to push both configs to, if any.

    Raises:
        ValueError: If no output target was given, or if the thresholds leave
            nothing behind.
    """
    from datasets import Dataset

    if not output_dir and not hub_id:
        raise ValueError("Give at least one of --out or --hub-id.")

    ds = Dataset.load_from_disk(judged)
    valid_col = f"{task_prefix}valid_fields"

    plain_rows: list[dict[str, Any]] = []
    reasoning_rows: list[dict[str, Any]] = []
    dropped: Counter = Counter()

    for row in ds:
        if not row.get(valid_col):
            dropped["judge returned no valid rubric"] += 1
            continue
        if not passes(row, min_score=min_score, min_grounding=min_grounding):
            dropped["below the rubric thresholds"] += 1
            continue

        scores = {field: row[field] for field in RUBRIC_FIELDS}
        shared = {key: row.get(key) for key in CARRIED_COLUMNS}
        shared["quality_score"] = round(
            sum(scores.values()) / len(RUBRIC_FIELDS), 3
        )
        answer = row["answer"]

        plain_rows.append(
            {
                **shared,
                **scores,
                "messages": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": answer},
                ],
            }
        )

        reasoning = row.get("reasoning") or ""
        if not reasoning:
            dropped["no reasoning trace (plain config only)"] += 1
            continue
        if row.get("reasoning_mentions_source") and not keep_source_mentions:
            dropped["trace refers to its source (plain config only)"] += 1
            continue

        reasoning_rows.append(
            {
                **shared,
                **scores,
                "thinking": reasoning,
                "content": answer,
                "messages": [
                    {"role": "user", "content": row["question"]},
                    {
                        "role": "assistant",
                        "content": (
                            f"<think>\n{reasoning}\n</think>\n\n{answer}"
                        ),
                    },
                ],
            }
        )

    stats = {
        "rows_judged": len(ds),
        "rows_plain": len(plain_rows),
        "rows_reasoning": len(reasoning_rows),
        "keep_rate": round(len(plain_rows) / len(ds), 4) if len(ds) else 0.0,
        "min_score": min_score,
        "min_grounding": min_grounding,
        "dropped": dict(dropped),
        "rubric_means": {
            field: round(
                sum(r[field] for r in plain_rows) / len(plain_rows), 3
            )
            for field in RUBRIC_FIELDS
        }
        if plain_rows
        else {},
        "question_types": dict(
            Counter(r["question_type"] for r in plain_rows)
        ),
    }

    print(
        f"{stats['rows_judged']:,} judged rows -> {stats['rows_plain']:,}"
        f" plain ({stats['keep_rate']:.1%}),"
        f" {stats['rows_reasoning']:,} with reasoning"
    )
    for reason, count in dropped.most_common():
        print(f"  dropped, {reason}: {count:,}")
    if stats["rubric_means"]:
        print("Rubric means of the kept rows:")
        for field, mean in stats["rubric_means"].items():
            print(f"  {field:24s} {mean:.2f}")

    if not plain_rows:
        raise ValueError(
            "The thresholds left no rows. Lower --min-score / --min-grounding,"
            " or read the judge output before blaming the filter."
        )

    configs = {
        "plain": Dataset.from_list(plain_rows),
        "reasoning": Dataset.from_list(reasoning_rows),
    }
    for name, config_ds in configs.items():
        if not len(config_ds):
            print(f"Skipping the empty '{name}' config")
            continue
        if output_dir:
            config_ds.save_to_disk(f"{output_dir}/{name}")
        if hub_id:
            config_ds.push_to_hub(hub_id, config_name=name)
            print(f"Pushed {hub_id} config '{name}'")

    if output_dir:
        stats_path = f"{output_dir}/sft_stats.json"
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)
        print(f"Wrote {output_dir}/{{plain,reasoning}} and {stats_path}")

    for config_ds in configs.values():
        config_ds.cleanup_cache_files()


def main(args: list[str] | None = None) -> None:
    """Build the SFT datasets from the command line.

    Args:
        args: Optional command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Filter the judged pairs on the rubric and write the plain and"
            " reasoning SFT configs."
        )
    )
    parser.add_argument(
        "--judged",
        default="examples/wiki-nl-persona-qa/outputs/judge/final",
        help="The judge pipeline's <output_dir>/final directory.",
    )
    parser.add_argument(
        "--task-prefix",
        default="judge-qa_",
        help=(
            "Task prefix the judge step wrote its bookkeeping columns under"
            " (default: judge-qa_, i.e. the step name in judge/pipeline.yaml)."
        ),
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=4,
        help="Lowest acceptable score on any rubric criterion.",
    )
    parser.add_argument(
        "--min-grounding",
        type=int,
        default=5,
        help=(
            "Lowest acceptable score on answer_correct and answer_grounded."
            " Stricter than --min-score because a wrong answer is worse"
            " training data than a stiff one."
        ),
    )
    parser.add_argument(
        "--keep-source-mentions",
        action="store_true",
        help=(
            "Keep reasoning traces that refer to the article they were shown."
            " Off by default: at inference time there is no article, so such a"
            " trace teaches the model to cite a source it does not have."
        ),
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        default=None,
        help="Directory to write the plain/ and reasoning/ configs under.",
    )
    parser.add_argument(
        "--hub-id",
        default=None,
        help="Hub dataset id; both configs are pushed to it by name.",
    )

    build_sft(**vars(parser.parse_args(args)))


if __name__ == "__main__":
    main()
