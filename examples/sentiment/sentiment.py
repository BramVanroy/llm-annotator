import random
import shutil

from llm_annotator import Annotator, VLLMOfflineClient
from llm_annotator.utils import get_hf_username


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Sentiment annotation on IMDB."
    )
    parser.add_argument(
        "--model", default="RedHatAI/gemma-3-27b-it-FP8-dynamic"
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--max-num-samples",
        type=int,
        default=200,
        help="Number of samples to process. Use -1 for the full dataset.",
    )
    parser.add_argument("--output-dir", default="outputs/sentiment-imdb-qwen")
    parser.add_argument(
        "--hub-id", default=None, help="HF Hub dataset ID to push to."
    )
    args = parser.parse_args(args)
    if args.max_num_samples == -1:
        args.max_num_samples = None

    hf_user = get_hf_username()
    prompt_prefix = """Analyze the sentiment of the following movie review and classify it as positive or negative.

Review:
"""  # noqa: W291
    prompt_template = (
        prompt_prefix
        + """{text}

Classification:"""
    )  # noqa: W291

    output_schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative"]}
        },
        "required": ["sentiment"],
    }

    def random_validitity(sample):
        return random.random() < 0.5

    def postprocess_fn(sample):
        # Example postprocessing: strip whitespace from sentiment
        if "sentiment" in sample and isinstance(sample["sentiment"], str):
            sample["sentiment"] = sample["sentiment"].strip()
        return sample

    client = VLLMOfflineClient(
        model=args.model,
        max_model_len=args.max_model_len,
    )
    hub_id = args.hub_id or (f"{hf_user}/sentiment-imdb" if hf_user else None)
    with Annotator(client=client, verbose=True) as anno:
        ds = anno.annotate_dataset(
            output_dir=args.output_dir,
            prompt_template=prompt_template,
            dataset_name="stanfordnlp/imdb",
            dataset_split="test",
            new_hub_id=hub_id,
            max_num_samples=args.max_num_samples,
            cache_input_dataset=False,  # `True` is generally useful, not for demo purposes
            output_schema=output_schema,
            keep_columns=["text", "label"],  # Keep all original columns
            # Backup to HF every 100 samples (in separate backup branch).
            # In practice, set to a higher value (e.g., 1000+)
            upload_every_n_samples=100,
            sort_by_length=True,  # Sort by prompt length for more efficient batching -- final dataset will be re-ordered to original
            # validate_fn=random_validitity,
            num_retries_invalid=3,
            postprocess_fn=postprocess_fn,
        )
    print(ds)
    shutil.rmtree(args.output_dir)


if __name__ == "__main__":
    main()
