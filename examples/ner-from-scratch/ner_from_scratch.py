from pathlib import Path

from llm_annotator import Annotator


CURR_DIR = Path(__file__).parent


def main():
    prompt = CURR_DIR.joinpath("prompt_template.md").read_text(encoding="utf-8")
    model = "RedHatAI/gemma-3-27b-it-FP8-dynamic"
    extra_vllm_init_kwargs = {"limit_mm_per_prompt": {"image": 0, "audio": 0}}
    sampling_params = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "max_tokens": 1024,
    }
    max_num_samples = 100

    with Annotator(
        model=model,
        verbose=True,
        max_model_len=2048,
        gpu_memory_utilization=0.95,
        extra_vllm_init_kwargs=extra_vllm_init_kwargs,
    ) as anno:
        anno.generate_dataset(
            output_dir=f"outputs/ner-from_scratch-{max_num_samples}",
            prompts=prompt,
            max_num_samples=max_num_samples,
            upload_every_n_samples=None,
            sampling_params=sampling_params,
        )


if __name__ == "__main__":
    main()
