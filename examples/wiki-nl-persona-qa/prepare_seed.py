"""Build the seed for the persona-grounded Dutch Wikipedia QA pipeline.

Samples clean Dutch Wikipedia articles and pairs each one with a random
Belgian-Dutch persona and a random question type. Both are what stop half a
million generated questions from sounding like the same person asking the same
kind of question about every article.

```sh
uv run examples/wiki-nl-persona-qa/prepare_seed.py \
    --num-samples 50000 --num-proc 16 \
    --out examples/wiki-nl-persona-qa/outputs/seed
```
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from collections.abc import Sequence


QUESTION_TYPES = {
    "feit": (
        "een vraag naar een concreet feit: een getal, datum, naam, plaats of"
        " hoeveelheid uit het artikel"
    ),
    "definitie": (
        "een vraag naar wat iets is of betekent, waarop het antwoord een"
        " omschrijving is"
    ),
    "uitleg": (
        "een waarom- of hoe-vraag, waarop het antwoord een mechanisme, reden"
        " of werkwijze uitlegt"
    ),
    "chronologie": (
        "een vraag naar de volgorde of het verloop van gebeurtenissen in de"
        " tijd"
    ),
    "vergelijking": (
        "een vraag die twee zaken uit het artikel tegenover elkaar zet, of"
        " vraagt waarin iets verschilt van iets anders"
    ),
    "gevolg": (
        "een vraag naar het gevolg, het effect of het belang van iets dat in"
        " het artikel beschreven wordt"
    ),
}
"""Question types to spread over the seed, as the Dutch instruction the prompt
renders into `{question_type}`. The value is written into the dataset, not the
key, so the prompt needs one placeholder rather than a lookup table."""


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
    """Sample finewiki-nl articles and attach a persona and a question type.

    Personas are drawn with replacement, so any `num_samples` works even though
    the persona pool holds 300k rows; the same persona then asks about several
    different articles, which is what the variability is for.

    Args:
        num_samples: Articles to keep.
        seed: Random seed for the shuffle, the persona draw and the type draw.
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

    personas = load_dataset(
        "nvidia/Nemotron-Personas-Belgium",
        split="nl_BE",
        num_proc=num_proc,
    ).select_columns(["persona"])
    print(f"{len(personas):,} Belgian-Dutch personas to draw from")

    rng = random.Random(seed)
    drawn = personas.select(
        [rng.randrange(len(personas)) for _ in range(len(ds))]
    )["persona"]
    types = rng.choices(list(QUESTION_TYPES.values()), k=len(ds))

    ds = ds.add_column("persona", drawn).add_column("question_type", types)
    print("Question types:", Counter(types).most_common())
    print(ds)

    ds.save_to_disk(output_dir)
    print(f"Wrote {len(ds):,} articles to {output_dir}")

    if hub_id:
        ds.push_to_hub(hub_id)
        print(f"Pushed to {hub_id}")

    ds.cleanup_cache_files()
    personas.cleanup_cache_files()


def main(args: list[str] | None = None) -> None:
    """Build the persona QA seed from the command line.

    Args:
        args: Optional command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sample Dutch Wikipedia articles and pair each with a random"
            " Belgian-Dutch persona and question type."
        )
    )
    parser.add_argument("--num-samples", type=int, default=50_000)
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
            "Longest article to keep (words), so article + persona + prompt"
            " + reasoning + answer fit the max_model_len the configs set."
        ),
    )
    parser.add_argument(
        "-j",
        "--num-proc",
        type=int,
        default=None,
        help=(
            "Processes to load and filter with. Default: unset, one process."
            " The output does not depend on this."
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
