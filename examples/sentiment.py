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
    """
    prompt_template = prompt_prefix + """{text}

    Classification:"""

    output_schema = {
        "type": "object",
        "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}},
        "required": ["sentiment"],
    }

    with Annotator(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=4096,
        verbose=True) as anno:
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
            # Backup to HF every 100 samples.
            # In practice, set to a higher value (e.g., 1000+)
            upload_every_n_samples=100,
        )
    print(ds)
    shutil.rmtree("outputs/sentiment-imdb-qwen")


if __name__ == "__main__":
    main()
