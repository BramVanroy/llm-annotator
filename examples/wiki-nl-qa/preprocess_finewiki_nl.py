from datasets import load_dataset

from llm_annotator.utils import get_hf_username


def main(num_workers: int | None = None):
    hf_user = get_hf_username()

    ds = load_dataset("HuggingFaceFW/finewiki", "nl")

    # Must contain three sentences (= three periods followed by a space, and three capital letters)
    # Must contain between 30 and 24,000 words
    # (Add space at the end to count the last sentence if it ends with a period)
    ds = ds.filter(
        lambda text: f"{text} ".count(". ") >= 3 and 30 < len(text.split()) < 24000,
        input_columns="text",
        num_proc=num_workers,
    )

    ds.push_to_hub(f"{hf_user}/finewiki-nl-30-to-24k-tokens")


if __name__ == "__main__":
    import argparse
    cparser = argparse.ArgumentParser()
    cparser.add_argument("-j", "--num-workers", type=int, default=None)
    cargs = cparser.parse_args()
    main(num_workers=cargs.num_workers)