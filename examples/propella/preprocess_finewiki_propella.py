from __future__ import annotations

from datasets import load_dataset


def is_stub(wikitext: str) -> bool:
    """Return ``True`` when a Wikipedia article is a stub page.

    Args:
        wikitext: Raw page text.

    Returns:
        Whether the page looks like a stub.
    """
    # E.g. overview pages like https://nl.wikipedia.org/wiki/Categorie:Wikipedia:Beginnetje_biologie
    return r"{{beginnetje" in wikitext.lower()


def main(args: list[str] | None = None) -> None:
    """Preprocess the finewiki source for the Propella example.

    Args:
        args: Optional command-line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess propella: truncate text to first 50,000 characters."
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument(
        "--hub-id",
        required=True,
        help="HF Hub dataset ID to push filtered dataset to.",
    )
    parsed_args = parser.parse_args(args)

    ds = load_dataset("HuggingFaceFW/finewiki", "nl")

    # Trim to 50,000 characters for propella
    ds = ds.filter(
        lambda wikitext: not is_stub(wikitext),
        input_columns=["wikitext"],
        num_proc=parsed_args.num_proc,
    ).map(
        lambda text: {"text_truncated": text[:50_000]},
        input_columns=["text"],
        num_proc=parsed_args.num_proc,
    )
    ds.push_to_hub(parsed_args.hub_id, private=True)


if __name__ == "__main__":
    main()
