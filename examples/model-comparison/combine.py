"""Stack the three per-model generation outputs into one judging dataset.

This script does two things:

1. Reports each model's generation-stage reliability: the schema-valid rate
   and error rate straight out of the annotator's own bookkeeping columns,
   before any judge sees a single row. A model that cannot reliably produce a
   well-formed ``{question, answer}`` object is unreliable before quality even
   enters the picture.
2. Concatenates the schema-valid rows into one long-format dataset (1,500 rows
   for 3 models x 500 articles), tagged with ``source_model``, ready for
   ``judge/pipeline.yaml`` to score in a single pass.

```sh
uv run examples/model-comparison/combine.py \
    --out examples/model-comparison/outputs/combined-qa
```
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any


DEFAULT_INPUTS = {
    "granite-4.1-3b": (
        "examples/model-comparison/outputs/generate/granite-4.1-3b/final"
    ),
    "granite-4.1-8b": (
        "examples/model-comparison/outputs/generate/granite-4.1-8b/final"
    ),
    "gemma-4-26b-a4b": (
        "examples/model-comparison/outputs/generate/gemma-4-26b-a4b/final"
    ),
}


def combine_outputs(
    *,
    inputs: dict[str, str],
    task_prefix: str,
    output_dir: str | None,
    hub_id: str | None,
) -> None:
    """Stack the per-model generation outputs and report their reliability.

    Args:
        inputs: Model name to its pipeline's ``<output_dir>/final`` directory.
        task_prefix: Prefix the generate step wrote its bookkeeping columns
            under.
        output_dir: Directory to write the combined dataset to, if any.
        hub_id: Hub dataset id to push the combined dataset to, if any.

    Raises:
        ValueError: If no output target was given, or if no valid row survived
            across any model.
    """
    from datasets import Dataset

    if not output_dir and not hub_id:
        raise ValueError("Give at least one of --out or --hub-id.")

    valid_col = f"{task_prefix}valid_fields"
    error_col = f"{task_prefix}error"
    finish_col = f"{task_prefix}finish_reason"
    tokens_col = f"{task_prefix}num_tokens"

    generation_stats: dict[str, dict[str, Any]] = {}
    combined_rows: list[dict[str, Any]] = []

    for model_name, path in inputs.items():
        ds = Dataset.load_from_disk(path)
        total = len(ds)
        num_valid = 0
        num_errors = 0
        finish_reasons: Counter = Counter()
        token_counts: list[int] = []

        for row in ds:
            finish_reasons[row.get(finish_col) or "unknown"] += 1
            if row.get(error_col):
                num_errors += 1
            num_tokens = row.get(tokens_col)
            if num_tokens:
                token_counts.append(num_tokens)

            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not row.get(valid_col) or not question or not answer:
                continue
            num_valid += 1
            combined_rows.append(
                {
                    "idx": row.get("idx"),
                    "title": row.get("title") or "",
                    "url": row.get("url") or "",
                    "text": row.get("text") or "",
                    "question": question,
                    "answer": answer,
                    "source_model": model_name,
                }
            )

        generation_stats[model_name] = {
            "rows_total": total,
            "rows_valid": num_valid,
            "valid_rate": round(num_valid / total, 4) if total else 0.0,
            "error_rate": round(num_errors / total, 4) if total else 0.0,
            "finish_reasons": dict(finish_reasons),
            "avg_num_tokens": (
                round(sum(token_counts) / len(token_counts), 1)
                if token_counts
                else 0.0
            ),
        }

    print("Generation-stage reliability (schema compliance, pre-judging):")
    print(f"  {'model':20s} {'rows':>6s} {'valid':>7s} {'errors':>7s}")
    for model_name, stats in generation_stats.items():
        print(
            f"  {model_name:20s} {stats['rows_total']:>6,}"
            f" {stats['valid_rate']:>6.1%} {stats['error_rate']:>6.1%}"
        )

    if not combined_rows:
        raise ValueError(
            "No valid generated question/answer pair survived across any"
            " model. The stats above say why."
        )

    combined = Dataset.from_list(combined_rows)
    print(
        f"\nCombined {len(combined):,} valid pairs from {len(inputs)} model(s)"
    )
    print("Rows per model:", Counter(r["source_model"] for r in combined_rows))

    if output_dir:
        combined.save_to_disk(output_dir)
        stats_path = f"{output_dir}_generation_stats.json"
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(generation_stats, fh, indent=2, ensure_ascii=False)
        print(f"Wrote {output_dir} and {stats_path}")
    if hub_id:
        combined.push_to_hub(hub_id)
        print(f"Pushed {hub_id}")

    combined.cleanup_cache_files()


def main(args: list[str] | None = None) -> None:
    """Combine the per-model generation outputs from the command line.

    Args:
        args: Optional command-line arguments.

    Raises:
        ValueError: If an ``--input`` is not of the form ``NAME=PATH``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Combine the per-model generation outputs into one dataset for"
            " the judge pipeline."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "One generation pipeline's <output_dir>/final directory, as"
            " NAME=PATH. Repeat for each model. Defaults to the three"
            " pipelines shipped in generate/."
        ),
    )
    parser.add_argument(
        "--task-prefix",
        default="generate-qa_",
        help=(
            "Task prefix the generate step wrote its bookkeeping columns"
            " under (default: generate-qa_, i.e. the step name in"
            " generate/pipeline-*.yaml)."
        ),
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        default=None,
        help="Directory to write the combined dataset.",
    )
    parser.add_argument("--hub-id", default=None, help="Hub dataset id.")
    parsed_args = parser.parse_args(args)

    inputs = dict(DEFAULT_INPUTS)
    for item in parsed_args.input:
        name, _, path = item.partition("=")
        if not name or not path:
            raise ValueError(f"--input must be NAME=PATH, got '{item}'.")
        inputs[name] = path

    combine_outputs(
        inputs=inputs,
        task_prefix=parsed_args.task_prefix,
        output_dir=parsed_args.output_dir,
        hub_id=parsed_args.hub_id,
    )


if __name__ == "__main__":
    main()
