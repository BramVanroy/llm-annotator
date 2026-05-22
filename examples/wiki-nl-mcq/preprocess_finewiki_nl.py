from datasets import load_dataset

from llm_annotator.utils import get_hf_username


def is_stub(wikitext: str):
    # Thanks to Edwin Rijgersberg for spotting this pattern of stubs!
    # E.g. overview pages like https://nl.wikipedia.org/wiki/Categorie:Wikipedia:Beginnetje_biologie
    return r"{{beginnetje" in wikitext.lower()


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter finewiki-nl for quality."
    )
    parser.add_argument("-j", "--num-workers", type=int, default=None)
    parser.add_argument(
        "--hub-id", default=None, help="HF Hub dataset ID to push to."
    )
    args = parser.parse_args(args)

    hf_user = get_hf_username()

    ds = load_dataset("HuggingFaceFW/finewiki", "nl")

    # Must contain three sentences (= three periods followed by a space, and three capital letters)
    # Must contain between 30 and 24,000 words
    # (Add space at the end to count the last sentence if it ends with a period)
    ds = ds.filter(
        lambda text, wikitext: (
            f"{text} ".count(". ") >= 3
            and 30 < len(text.split()) < 24_000
            and 200 < len(text) < 72_000
            and not is_stub(wikitext)
        ),
        input_columns=["text", "wikitext"],
        num_proc=args.num_workers,
    )
    print(ds)

    hub_id = args.hub_id or (
        f"{hf_user}/finewiki-nl-30-to-24k-tokens" if hf_user else None
    )
    if hub_id:
        ds.push_to_hub(hub_id)


if __name__ == "__main__":
    main()
