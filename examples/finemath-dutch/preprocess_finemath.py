from __future__ import annotations

from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from llm_annotator.utils import get_hf_username


CURR_DIR = Path(__file__).parent


def main(args: list[str] | None = None) -> None:
    """Preprocess finemath for the Dutch translation example.

    Args:
        args: Optional command-line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess finemath: count tokens and filter by length."
    )
    parser.add_argument(
        "--tokenizer",
        default="RedHatAI/gemma-3-27b-it-FP8-dynamic",
        help="Tokenizer model to use for token counting.",
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceTB/finemath",
        help="Dataset name on HF Hub.",
    )
    parser.add_argument("--dataset-config", default="finemath-4plus")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--token-limit",
        type=int,
        default=36_000,
        help="Maximum number of tokens allowed per sample.",
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument(
        "--hub-id",
        default=None,
        help="HF Hub dataset ID to push filtered dataset to.",
    )
    parsed_args = parser.parse_args(args)

    hf_user = get_hf_username()

    prompt_template = CURR_DIR.joinpath("prompt_template.md").read_text(
        encoding="utf-8"
    )
    ds = load_dataset(
        parsed_args.dataset,
        parsed_args.dataset_config,
        split=parsed_args.dataset_split,
    )
    tokenizer = AutoTokenizer.from_pretrained(parsed_args.tokenizer)

    def count_tokens_with_template(texts):
        prompted_texts = [
            [{"role": "user", "content": prompt_template.format(text=text)}]
            for text in texts
        ]
        tokens = [
            tokenizer.apply_chat_template(
                prompted_text, encode=True, add_generation_prompt=True
            )
            for prompted_text in prompted_texts
        ]
        return {"num_tokens": [len(ids) for ids in tokens]}

    ds = ds.map(
        count_tokens_with_template,
        batched=True,
        batch_size=1000,
        input_columns=["text"],
        num_proc=parsed_args.num_proc,
    )

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

    ds = ds.filter(
        lambda num_tokens: num_tokens <= parsed_args.token_limit,
        input_columns=["num_tokens"],
        num_proc=parsed_args.num_proc,
    ).remove_columns(["num_tokens"])

    hub_id = parsed_args.hub_id or (
        f"{hf_user}/finemath-4plus-lte{parsed_args.token_limit}"
        if hf_user
        else None
    )
    if hub_id:
        ds.push_to_hub(hub_id, private=False)


if __name__ == "__main__":
    main()
