from datasets import load_dataset

from llm_annotator.utils import get_hf_username


def main():
    hf_user = get_hf_username()

    ds = load_dataset("HuggingFaceFW/finewiki", "nl")

    # Must contain three sentences (= three periods followed by a space, and three capital letters)
    # Must contain more than 200 characters
    # (Add space at the end to count the last sentence if it ends with a period)
    ds = ds.filter(
        lambda text: f"{text} ".count(". ") >= 3 and len(text) > 200,
        input_columns="text",
        num_proc=64,
    )

    ds.push_to_hub(f"{hf_user}/finewiki-nl-gte200")


if __name__ == "__main__":
    main()
