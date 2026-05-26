from datasets import load_dataset


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess propella: truncate text to first 50,000 characters."
    )
    parser.add_argument("--dataset", help="Dataset name on HF Hub.")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--text-column",
        default="text",
        help="Dataset column containing input text.",
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument(
        "--hub-id",
        default=None,
        help="HF Hub dataset ID to push filtered dataset to.",
    )
    args = parser.parse_args(args)

    def count_tokens_with_template(texts):
        truncated_texts = [text[:50_000] for text in texts]
        return {f"{args.text_column}_truncated": truncated_texts}

    ds = load_dataset(
        args.dataset, args.dataset_config, split=args.dataset_split
    )
    ds = ds.map(
        count_tokens_with_template,
        batched=True,
        batch_size=10000,
        input_columns=[args.text_column],
        num_proc=args.num_proc,
    )
    ds.push_to_hub(args.hub_id, private=False)


if __name__ == "__main__":
    main()
