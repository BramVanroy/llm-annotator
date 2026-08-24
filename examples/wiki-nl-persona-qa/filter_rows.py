"""Drop the generated pairs that are unusable as training data.

The answer step is served with `--reasoning-parser qwen3`, so its trace already
arrives in its own `answer-question_reasoning` column and the answer alone is in
`answer-question_response`. This script decides which of those rows are worth
judging, and writes the dataset the judge pipeline reads.

```sh
uv run --frozen examples/wiki-nl-persona-qa/filter_rows.py \
    --generated examples/wiki-nl-persona-qa/outputs/generate/final \
    --out examples/wiki-nl-persona-qa/outputs/qa-split
```
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from typing import Any


SOURCE_PHRASES = (
    "dit artikel",
    "het artikel",
    "deze tekst",
    "de tekst",
    "het fragment",
    "de passage",
    "hierboven",
    "bovenstaand",
    "de bron",
    "het bronmateriaal",
    "dit stuk",
)
"""Dutch phrases that give away that the model is quoting the article it was
shown. Trained on, they teach a model to cite a source it will not have at
inference time. The list is deliberately over-eager -- "de tekst" also fires on
a song's lyrics -- because losing a few good rows is cheaper than shipping rows
that reference a document the reader never saw."""

SOURCE_PATTERN = re.compile(
    "|".join(re.escape(phrase) for phrase in SOURCE_PHRASES),
    flags=re.IGNORECASE,
)


def filter_generated(
    *,
    generated: str,
    task_prefix: str,
    keep_truncated: bool,
    output_dir: str | None,
    hub_id: str | None,
) -> None:
    """Filter the generation output and report what survived.

    Args:
        generated: The generate pipeline's ``<output_dir>/final`` directory.
        task_prefix: Prefix the answer step wrote its bookkeeping columns
            under.
        keep_truncated: Keep rows whose answer hit the token budget instead of
            dropping them.
        output_dir: Directory to write the filtered dataset to, if any.
        hub_id: Hub dataset id to push the filtered dataset to, if any.

    Raises:
        ValueError: If no output target was given, or if no row survived.
    """
    from datasets import Dataset

    if not output_dir and not hub_id:
        raise ValueError("Give at least one of --out or --hub-id.")

    response_col = f"{task_prefix}response"
    reasoning_col = f"{task_prefix}reasoning"
    error_col = f"{task_prefix}error"
    finish_col = f"{task_prefix}finish_reason"
    tokens_col = f"{task_prefix}num_tokens"

    ds = Dataset.load_from_disk(generated)
    dropped: Counter = Counter()
    finish_reasons: Counter = Counter()
    rows: list[dict[str, Any]] = []

    for row in ds:
        finish_reasons[row.get(finish_col) or "unknown"] += 1

        if row.get(error_col):
            dropped["provider error"] += 1
            continue
        if row.get(finish_col) == "length" and not keep_truncated:
            dropped["truncated at max_completion_tokens"] += 1
            continue

        question = (row.get("question") or "").strip()
        answer = (row.get(response_col) or "").strip()
        reasoning = (row.get(reasoning_col) or "").strip()
        if not question or not answer:
            dropped["empty question or answer"] += 1
            continue
        if SOURCE_PATTERN.search(answer):
            dropped["answer refers to its source"] += 1
            continue

        rows.append(
            {
                "idx": row.get("idx"),
                "title": row.get("title") or "",
                "url": row.get("url") or "",
                "text": row.get("text") or "",
                "persona": row.get("persona") or "",
                "question_type": row.get("question_type") or "",
                "question": question,
                "reasoning": reasoning,
                "answer": answer,
                "has_reasoning": bool(reasoning),
                "reasoning_mentions_source": bool(
                    reasoning and SOURCE_PATTERN.search(reasoning)
                ),
                "num_tokens": row.get(tokens_col),
            }
        )

    stats = {
        "rows_in": len(ds),
        "rows_out": len(rows),
        "keep_rate": round(len(rows) / len(ds), 4) if len(ds) else 0.0,
        "with_reasoning": sum(r["has_reasoning"] for r in rows),
        "reasoning_mentions_source": sum(
            r["reasoning_mentions_source"] for r in rows
        ),
        "dropped": dict(dropped),
        "finish_reasons": dict(finish_reasons),
    }

    print(f"{stats['rows_in']:,} generated rows -> {stats['rows_out']:,} kept")
    for reason, count in dropped.most_common():
        print(f"  dropped, {reason}: {count:,}")
    print(
        f"  with a reasoning trace: {stats['with_reasoning']:,}"
        f" ({stats['reasoning_mentions_source']:,} of them refer to a source"
        " and are excluded from the reasoning SFT config)"
    )
    if rows and not stats["with_reasoning"]:
        print(
            "  No row has a trace at all. Either the server ran without"
            " --reasoning-parser qwen3, or the step ran with thinking off."
        )

    if not rows:
        raise ValueError(
            "No usable question/answer pair survived. The counts above say"
            " why; a wall of 'truncated' means max_completion_tokens is too"
            " small for the reasoning trace."
        )

    filtered = Dataset.from_list(rows)
    if output_dir:
        filtered.save_to_disk(output_dir)
        stats_path = f"{output_dir}_filter_stats.json"
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)
        print(f"Wrote {output_dir} and {stats_path}")
    if hub_id:
        filtered.push_to_hub(hub_id)
        print(f"Pushed {hub_id}")

    filtered.cleanup_cache_files()


def main(args: list[str] | None = None) -> None:
    """Filter the generation output from the command line.

    Args:
        args: Optional command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Drop the generated question/answer pairs that are unusable as"
            " training data, and write the dataset the judge reads."
        )
    )
    parser.add_argument(
        "--generated",
        default="examples/wiki-nl-persona-qa/outputs/generate/final",
        help="The generate pipeline's <output_dir>/final directory.",
    )
    parser.add_argument(
        "--task-prefix",
        default="answer-question_",
        help=(
            "Task prefix the answer step wrote its bookkeeping columns under"
            " (default: answer-question_, i.e. the step name in"
            " generate/pipeline-qa.yaml)."
        ),
    )
    parser.add_argument(
        "--keep-truncated",
        action="store_true",
        help=(
            "Keep rows that hit max_completion_tokens. Off by default: an"
            " answer cut mid-sentence is broken training data."
        ),
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        default=None,
        help="Directory to write the filtered dataset to.",
    )
    parser.add_argument("--hub-id", default=None, help="Hub dataset id.")

    filter_generated(**vars(parser.parse_args(args)))


if __name__ == "__main__":
    main()
