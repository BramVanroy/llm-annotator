from transformers import AutoTokenizer
from llm_annotator.annotator import load_dataset
from llm_annotator.utils import get_hf_username
from pathlib import Path

import numpy as np

CURR_DIR = Path(__file__).parent


def main():
    hf_user = get_hf_username()

    prompt_template = CURR_DIR.joinpath("prompt_template.md").read_text(encoding="utf-8")
    ds = load_dataset("HuggingFaceTB/finemath", "finemath-4plus", split="train")
    tokenizer = AutoTokenizer.from_pretrained("RedHatAI/gemma-3-27b-it-FP8-dynamic")

    def count_tokens_with_template(texts):
        prompted_texts = [[{"role": "user", "content": prompt_template.format(text=text)}] for text in texts]
        tokens = [tokenizer.apply_chat_template(prompted_text, encode=True, add_generation_prompt=True) for prompted_text in prompted_texts]
        return {"num_tokens": [len(ids) for ids in tokens]}

    ds = ds.map(count_tokens_with_template, batched=True, batch_size=1000, input_columns=["text"], num_proc=8)

    # Print distribution of token counts (e.g., mean, min, max, 95th percentile, etc.)
    token_counts = np.array(ds["num_tokens"])
    print("Token count statistics:")
    print(f"Mean: {np.mean(token_counts)}")
    print(f"Min: {np.min(token_counts)}")
    print(f"Max: {np.max(token_counts)}")
    print(f"95th percentile: {np.percentile(token_counts, 95)}")
    print(f"99th percentile: {np.percentile(token_counts, 99)}")
    print(f"99.5th percentile: {np.percentile(token_counts, 99.5)}")
    print(f"99.9th percentile: {np.percentile(token_counts, 99.9)}")
    print(f"99.95th percentile: {np.percentile(token_counts, 99.95)}")
    print(f"99.99th percentile: {np.percentile(token_counts, 99.99)}")

    ds = ds.filter(lambda num_tokens: num_tokens <= 36000, input_columns=["num_tokens"], num_proc=8).remove_columns(["num_tokens"])
    ds.push_to_hub(f"{hf_user}/finemath-4plus-lte36k", private=False)


if __name__ == "__main__":
    main()