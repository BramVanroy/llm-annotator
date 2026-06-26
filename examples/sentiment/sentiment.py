from __future__ import annotations

import random
import shutil

from llm_annotator import Annotator, VLLMOfflineClient
from llm_annotator.utils import get_hf_username


def main(args: list[str] | None = None) -> None:
    """Run the sentiment annotation example.

    Args:
        args: Optional command-line arguments.
    """
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
        default=20,
        help="Number of samples to process. Use -1 for the full dataset.",
    )
    parser.add_argument("--output-dir", default="outputs/sentiment-imdb-qwen")
    parser.add_argument(
        "--hub-id", default=None, help="HF Hub dataset ID to push to."
    )
    parser.add_argument(
        "--force-data-preparation",
        action="store_true",
        default=False,
        help="Force regeneration of prepared data.",
    )
    parsed_args = parser.parse_args(args)
    if parsed_args.max_num_samples == -1:
        parsed_args.max_num_samples = None

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

    # You can add a function that determines whether output was valid
    # The column `valid` highlights which samples were valid
    def random_validitity(sample):
        return random.random() < 0.5

    def postprocess_fn(sample):
        # Example postprocessing: strip whitespace from sentiment
        if "sentiment" in sample and isinstance(sample["sentiment"], str):
            sample["sentiment"] = sample["sentiment"].strip()
        return sample

    client = VLLMOfflineClient(
        model=parsed_args.model,
        max_model_len=parsed_args.max_model_len,
    )
    hub_id = parsed_args.hub_id or (
        f"{hf_user}/sentiment-imdb" if hf_user else None
    )
    with Annotator(client=client, verbose=True) as anno:
        prepared_dataset, _, _ = anno.prepare_data(
            output_dir=parsed_args.output_dir,
            prompt_template=prompt_template,
            dataset_name="stanfordnlp/imdb",
            dataset_split="test",
            max_num_samples=parsed_args.max_num_samples,
            sort_by_length=True,  # Sort by prompt length for more efficient batching -- final dataset will be re-ordered to original
            hub_id=hub_id,
            force_data_preparation=parsed_args.force_data_preparation,
        )

        ds = anno.run_annotation(
            output_dir=parsed_args.output_dir,
            prompt_template=prompt_template,
            prepared_dataset=prepared_dataset,
            hub_id=hub_id,
            output_schema=output_schema,
            keep_columns=["text", "label"],  # Keep all original columns
            num_retries_invalid=3,
            postprocess_fn=postprocess_fn,
            # validate_fn=random_validitity,
        )
    print(ds)
    shutil.rmtree(parsed_args.output_dir)


if __name__ == "__main__":
    main()
