from pathlib import Path

from llm_annotator import (
    Annotator,
    VLLMOfflineClient,
    VLLMRuntimeOptions,
)


CURR_DIR = Path(__file__).parent


def main():
    prompt = CURR_DIR.joinpath("prompt_template.md").read_text(
        encoding="utf-8"
    )
    model = "RedHatAI/gemma-3-27b-it-FP8-dynamic"
    extra_vllm_kwargs = {"limit_mm_per_prompt": {"image": 0, "audio": 0}}
    options = VLLMRuntimeOptions(
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_tokens=1024,
    )
    max_num_samples = 100

    client = VLLMOfflineClient(
        model=model,
        max_model_len=2048,
        gpu_memory_utilization=0.95,
        extra_vllm_kwargs=extra_vllm_kwargs,
    )
    with Annotator(client=client, verbose=True) as anno:
        anno.generate_dataset(
            output_dir=f"outputs/ner-from_scratch-{max_num_samples}",
            prompts=prompt,
            max_num_samples=max_num_samples,
            upload_every_n_samples=None,
            options=options,
        )


if __name__ == "__main__":
    main()
