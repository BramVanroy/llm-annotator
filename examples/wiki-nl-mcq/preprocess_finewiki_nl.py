from datasets import load_dataset

from llm_annotator.utils import get_hf_username


def is_stub(wikitext: str):
    # Thanks to Edwin Rijgersberg for spotting this pattern of stubs!
    # E.g. overview pages like https://nl.wikipedia.org/wiki/Categorie:Wikipedia:Beginnetje_biologie
    return r"{{beginnetje" in wikitext.lower()


def main(num_workers: int | None = None):
    hf_user = get_hf_username()

    ds = load_dataset("HuggingFaceFW/finewiki", "nl")

    # Must contain three sentences (= three periods followed by a space, and three capital letters)
    # Must contain between 30 and 24,000 words
    # (Add space at the end to count the last sentence if it ends with a period)
    ds = ds.filter(
        lambda text, wikitext: f"{text} ".count(". ") >= 3
        and 30 < len(text.split()) < 24_000
        and 200 < len(text) < 72_000
        and not is_stub(wikitext),
        input_columns=["text", "wikitext"],
        num_proc=num_workers,
    )
    print(ds)

    ds.push_to_hub(f"{hf_user}/finewiki-nl-30-to-24k-tokens")


if __name__ == "__main__":
    import argparse

    cparser = argparse.ArgumentParser()
    cparser.add_argument("-j", "--num-workers", type=int, default=None)
    cargs = cparser.parse_args()
    main(num_workers=cargs.num_workers)
