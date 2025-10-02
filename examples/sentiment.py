
import json
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
    prompt_template = """Analyze the sentiment of the following movie review and classify it as positive or negative.

    Review: {text}

    Classification:"""

    output_schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"]
            }
        },
        "required": ["sentiment"]
    }

    anno = Annotator(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        prompt_template=prompt_template,
        output_schema=output_schema,
        # Backup to HF every 10 samples.
        # In practice, set to a higher value (e.g., 1000+)
        upload_every_n_samples=10,
    )
    ds = anno.annotate_dataset(
        "stanfordnlp/imdb",
        output_dir="outputs/sentiment-imdb-qwen",
        dataset_split="test",
        new_hub_id=f"{hf_user}/sentiment-imdb",
        streaming=True,
        max_num_samples=100,
        cache_input_dataset=False,  # `True` is generally useful, not for demo purposes
    )
    print(ds)

if __name__ == "__main__":
    main()