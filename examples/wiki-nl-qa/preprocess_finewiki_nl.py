from datasets import load_dataset

from llm_annotator.utils import get_hf_username


def main():
    hf_user = get_hf_username()

    ds = load_dataset("HuggingFaceFW/finewiki", "nl")

    # Must contain three sentences (= three periods followed by a space, and three capital letters)
    # Must contain between 30 and 24,000 words
    # (Add space at the end to count the last sentence if it ends with a period)
    ds = ds.filter(
        lambda text: f"{text} ".count(". ") >= 3 and 30 < len(text.split()) < 24000,
        input_columns="text",
        num_proc=8,
    )

    ds.push_to_hub(f"{hf_user}/finewiki-nl-30-to-24k-tokens")


if __name__ == "__main__":
    main()
