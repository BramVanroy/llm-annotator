"""Build a small, clean seed of Dutch Wikipedia articles for model comparison.

```sh
uv run examples/model-comparison/prepare_seed.py \
    --num-samples 500 --num-proc 16 \
    --out examples/model-comparison/outputs/seed
```
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def keep_articles(
    texts: Sequence[str],
    *,
    min_words: int,
    max_words: int,
) -> list[bool]:
    """Apply the quality filter to a batch of articles.

    Naive checking of how many sentences (`. `) >= 3 and word count
    in the range (min_words, max_words).

    Args:
        texts: Rendered article texts.
        min_words: Shortest article to keep, exclusive.
        max_words: Longest article to keep, exclusive.

    Returns:
        One keep/drop flag per article, in input order.
    """
    return [
        text.count(". ") + int(text.endswith(".")) >= 3
        and min_words < len(text.split()) < max_words
        for text in texts
    ]


def build_seed(
    *,
    num_samples: int,
    seed: int,
    min_words: int,
    max_words: int,
    num_proc: int | None,
    output_dir: str,
    hub_id: str | None,
) -> None:
    """Filter and sample the finewiki-nl seed for the model comparison.

    Args:
        num_samples: Articles to keep, shared by every generator.
        seed: Random seed for the shuffle before sampling.
        min_words: Shortest article to keep, exclusive.
        max_words: Longest article to keep, exclusive.
        num_proc: Processes to load and filter with, or ``None`` for one.
        output_dir: Directory to write the seed dataset to.
        hub_id: Hub dataset id to push to, if any.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/finewiki",
        "nl",
        num_proc=num_proc,
        split="train",
    )

    ds = ds.filter(
        keep_articles,
        input_columns=["text"],
        batched=True,
        fn_kwargs={"min_words": min_words, "max_words": max_words},
        num_proc=num_proc,
    )
    print(f"{len(ds):,} articles pass the quality filter")

    ds = (
        ds.shuffle(seed=seed)
        .select_columns(["title", "text", "url"])
        .select(range(min(num_samples, len(ds))))
    )
    print(ds)

    ds.save_to_disk(output_dir)
    print(f"Wrote {len(ds):,} articles to {output_dir}")

    if hub_id:
        ds.push_to_hub(hub_id)
        print(f"Pushed to {hub_id}")

    ds.cleanup_cache_files()


def main(args: list[str] | None = None) -> None:
    """Build the model-comparison seed from the command line.

    Args:
        args: Optional command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sample a clean seed of Dutch Wikipedia articles for the"
            " model-comparison pipeline."
        )
    )
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-words",
        type=int,
        default=50,
        help="Shortest article to keep (words).",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=2560,
        help=(
            "Longest article to keep (words), so title + article + prompt"
            " + answer fit one context window without raising"
            " MAX_MODEL_LEN."
        ),
    )
    parser.add_argument(
        "-j",
        "--num-proc",
        type=int,
        default=None,
        help=(
            "Processes to filter with. Default: unset, one"
            " process. The output does not depend on this."
        ),
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        required=True,
        help="Local directory to write the seed dataset to.",
    )
    parser.add_argument(
        "--hub-id",
        default=None,
        help="Optional HF Hub dataset id to push to.",
    )

    build_seed(**vars(parser.parse_args(args)))


if __name__ == "__main__":
    main()
