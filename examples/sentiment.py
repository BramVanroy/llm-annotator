import random
import shutil

from huggingface_hub import HfApi

from llm_annotator import Annotator


def get_hf_username() -> str | None:
    whoami = HfApi().whoami()
    if whoami and "name" in whoami and whoami["type"] == "user":
        return whoami["name"]
    else:
        raise ValueError("No Hugging Face username found. Please login using `hf auth login`.")


def main():
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
        "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative"]}},
        "required": ["sentiment"],
    }

    def random_validitity(sample):
        return random.random() < 0.5

    def postprocess_fn(sample):
        # Example postprocessing: strip whitespace from sentiment
        if "sentiment" in sample and isinstance(sample["sentiment"], str):
            sample["sentiment"] = sample["sentiment"].strip()
        return sample

    model = "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8"
    extra_vllm_init_kwargs = {"tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral"}

    model = "RedHatAI/gemma-3-27b-it-FP8-dynamic"
    extra_vllm_init_kwargs = {}
    with Annotator(
        model=model, max_model_len=4096, verbose=True, extra_vllm_init_kwargs=extra_vllm_init_kwargs
    ) as anno:
        ds = anno.annotate_dataset(
            output_dir="outputs/sentiment-imdb-qwen",
            full_prompt_template=prompt_template,
            dataset_name="stanfordnlp/imdb",
            dataset_split="test",
            new_hub_id=f"{hf_user}/sentiment-imdb",
            streaming=True,
            max_num_samples=200,
            cache_input_dataset=False,  # `True` is generally useful, not for demo purposes
            prompt_template_prefix=prompt_prefix,
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
    shutil.rmtree("outputs/sentiment-imdb-qwen")


if __name__ == "__main__":
    main()
