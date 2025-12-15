import json
from pathlib import Path

from llm_annotator import Annotator
from llm_annotator.utils import get_hf_username


CURR_DIR = Path(__file__).parent


def main():
    hf_user = get_hf_username()
    prompt_template = CURR_DIR.joinpath("prompt_template.md").read_text(encoding="utf-8")
    system_message = CURR_DIR.joinpath("system_prompt.md").read_text(encoding="utf-8")
    json_schema = json.loads(CURR_DIR.joinpath("output.schema.json").read_text(encoding="utf-8"))

    model = "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8"
    dataset = "BramVanroy/finewiki-nl-30-to-24k-tokens"

    sampling_params = {
        "temperature": 0.15,
        "max_tokens": 12000,
    }
    extra_vllm_init_kwargs = {
        "load_format": "mistral",
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "limit_mm_per_prompt": {"image": 0},
    }
    with Annotator(
        model=model,
        verbose=True,
        max_model_len=48000,
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        extra_vllm_init_kwargs=extra_vllm_init_kwargs,
        num_proc=8,
    ) as anno:
        anno.annotate_dataset(
            output_dir="outputs/wiki-nl-mcq",
            prompt_template=prompt_template,
            dataset_name=dataset,
            dataset_split="train",
            new_hub_id=f"{hf_user}/wiki-nl-mcq",
            keep_columns=["text", "title", "url"],
            upload_every_n_samples=None,
            sampling_params=sampling_params,
            system_message=system_message,
            sort_by_length=True,
            output_schema=json_schema,

        )


if __name__ == "__main__":
    main()
