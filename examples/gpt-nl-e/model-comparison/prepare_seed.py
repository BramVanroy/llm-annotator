"""Build a small, clean seed of Dutch Wikipedia articles for model comparison.

Filters ``HuggingFaceFW/finewiki`` (``nl`` config) down to non-stub articles
short enough that title + article + prompt + a generated answer comfortably
fit one context window, then samples ``--num-samples`` of them. All three
``generate/pipeline-*.yaml`` configs read the same seed, so the three models
are compared on exactly the same 200 articles.

```sh
uv run examples/gpt-nl-e/model-comparison/prepare_seed.py \
    --num-samples 200 \
    --out examples/gpt-nl-e/model-comparison/outputs/seed
```
"""

from __future__ import annotations

import argparse


def is_stub(wikitext: str) -> bool:
    """Return ``True`` when a Wikipedia article is a stub page.

    Args:
        wikitext: Raw page text.

    Returns:
        Whether the page looks like a stub.
    """
    # Thanks to Edwin Rijgersberg for spotting this pattern of stubs!
    return r"{{beginnetje" in wikitext.lower()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sample a clean seed of Dutch Wikipedia articles for the"
            " model-comparison pipeline."
        )
    )
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-words",
        type=int,
        default=60,
        help="Shortest article to keep (words).",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=2000,
        help=(
            "Longest article to keep (words), so title + article + prompt"
            " + answer fit one context window without raising"
            " MAX_MODEL_LEN."
        ),
    )
    parser.add_argument("-j", "--num-workers", type=int, default=None)
    parser.add_argument(
        "--out",
        default="examples/gpt-nl-e/model-comparison/outputs/seed",
        help="Local directory to write the seed dataset to.",
    )
    parser.add_argument(
        "--hub-id",
        default=None,
        help="Optional HF Hub dataset id to push to.",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    """Filter and sample the finewiki-nl seed for the model-comparison pipeline.

    Args:
        args: Optional command-line arguments.
    """
    from datasets import load_dataset

    parsed_args = build_parser().parse_args(args)

    ds = load_dataset("HuggingFaceFW/finewiki", "nl", split="train")

    # Same three checks as the wiki-nl-mcq example, plus a tighter upper
    # bound: MCQ generation reads the whole article as multiple-choice
    # source material, but a single grounded QA pair does not need it, and
    # a shorter article keeps every model's generate step inside the
    # default MAX_MODEL_LEN.
    ds = ds.filter(
        lambda text, wikitext: (
            f"{text} ".count(". ") >= 3
            and parsed_args.min_words
            < len(text.split())
            < parsed_args.max_words
            and not is_stub(wikitext)
        ),
        input_columns=["text", "wikitext"],
        num_proc=parsed_args.num_workers,
    )
    print(f"{len(ds):,} articles pass the quality filter")

    ds = ds.shuffle(seed=parsed_args.seed)
    ds = ds.select(range(min(parsed_args.num_samples, len(ds))))
    ds = ds.select_columns(["title", "text", "url"])
    print(ds)

    ds.save_to_disk(parsed_args.out)
    print(f"Wrote {len(ds):,} articles to {parsed_args.out}")

    if parsed_args.hub_id:
        ds.push_to_hub(parsed_args.hub_id)
        print(f"Pushed to {parsed_args.hub_id}")


if __name__ == "__main__":
    main()
