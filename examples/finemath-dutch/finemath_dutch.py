from pathlib import Path

from llm_annotator import Annotator
from llm_annotator.utils import get_hf_username


CURR_DIR = Path(__file__).parent


def main():
    hf_user = get_hf_username()
    prompt_prefix = CURR_DIR.joinpath("prompt_prefix.md").read_text(encoding="utf-8")
    prompt_template = CURR_DIR.joinpath("prompt_template.md").read_text(encoding="utf-8")

    model = "RedHatAI/gemma-3-27b-it-FP8-dynamic"
    extra_vllm_init_kwargs = {
        "limit_mm_per_prompt": {"image": 0, "audio": 0},
    }
    # Also change "upload every_n_samples" below if you change this.
    max_num_samples = 10
    sampling_params = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 36_000,
    }
    with Annotator(
        model=model,
        verbose=True,
        max_model_len=72_000,
        max_num_seqs=4,
        gpu_memory_utilization=0.95,
        extra_vllm_init_kwargs=extra_vllm_init_kwargs
    ) as anno:
        anno.annotate_dataset(
            output_dir=f"outputs/finemath-dutch-{max_num_samples}",
            full_prompt_template=prompt_template,
            dataset_name="BramVanroy/finemath-4plus-seqlen36k",
            dataset_split="train",
            new_hub_id=f"{hf_user}/finemath-dutch-{max_num_samples}",
            max_num_samples=max_num_samples,
            prompt_template_prefix=prompt_prefix,
            keep_columns=True,
            upload_every_n_samples=None,
            sampling_params=sampling_params,
        )


if __name__ == "__main__":
    main()
