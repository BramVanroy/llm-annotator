from datasets import load_dataset


def is_stub(wikitext: str) -> bool:
    # E.g. overview pages like https://nl.wikipedia.org/wiki/Categorie:Wikipedia:Beginnetje_biologie
    return r"{{beginnetje" in wikitext.lower()


def main(args=None):
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
    args = parser.parse_args(args)

    ds = load_dataset("HuggingFaceFW/finewiki", "nl")

    # Trim to 50,000 characters for propella
    ds = ds.filter(
        lambda wikitext: not is_stub(wikitext),
        input_columns=["wikitext"],
        num_proc=args.num_proc,
    ).map(
        lambda text: {"text_truncated": text[:50_000]},
        input_columns=["text"],
        num_proc=args.num_proc,
    )
    ds.push_to_hub(args.hub_id, private=True)


if __name__ == "__main__":
    main()
