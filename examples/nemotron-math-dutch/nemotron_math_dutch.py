from pathlib import Path

from llm_annotator import Annotator


CURR_DIR = Path(__file__).parent


def main():
    prompt_prefix = CURR_DIR.joinpath("prompt_prefix.md").read_text(encoding="utf-8")
    prompt_template = CURR_DIR.joinpath("prompt_template.md").read_text(encoding="utf-8")

    model = "RedHatAI/gemma-3-27b-it-FP8-dynamic"
    extra_vllm_init_kwargs = {"limit_mm_per_prompt": {"image": 0, "audio": 0}}
    max_num_samples = 10_000
    sampling_params = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
    }
    with Annotator(
        model=model, verbose=True, max_model_len=64_000, extra_vllm_init_kwargs=extra_vllm_init_kwargs
    ) as anno:
        anno.annotate_dataset(
            output_dir=f"outputs/nemotron-math-dutch-{max_num_samples}",
            full_prompt_template=prompt_template,
            dataset_name="nvidia/Nemotron-CC-Math-v1",
            dataset_config="4plus_MIND",
            dataset_split="train",
            streaming=True,
            new_hub_id=f"BramVanroy/Nemotron-CC-Math-v1-dutch-{max_num_samples}",
            max_num_samples=max_num_samples,
            prompt_template_prefix=prompt_prefix,
            keep_columns=True,
            upload_every_n_samples=None,
            sort_by_length=True,
            sampling_params=sampling_params,
        )


if __name__ == "__main__":
    main()
