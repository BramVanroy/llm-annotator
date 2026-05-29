from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from llm_annotator import (
    Annotator,
    VLLMOfflineClient,
    VLLMOfflineRuntimeOptions,
)


CURR_DIR = Path(__file__).parent


def main(args: list[str] | None = None) -> None:
    """Generate a NER dataset from scratch.

    Args:
        args: Optional command-line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate NER dataset from scratch."
    )
    parser.add_argument(
        "--model", default="RedHatAI/gemma-3-27b-it-FP8-dynamic"
    )
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--max-num-samples",
        type=int,
        default=100,
        help="Number of samples to generate. Use -1 for no limit.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to outputs/ner-from_scratch-{n}.",
    )
    parser.add_argument(
        "--hub-id",
        default=None,
        help="Optional HF Hub dataset ID for prepared and generated artifacts.",
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

    prompt = CURR_DIR.joinpath("prompt_template.md").read_text(
        encoding="utf-8"
    )
    n = parsed_args.max_num_samples
    n_str = str(n) if n is not None else "full"
    output_dir = parsed_args.output_dir or f"outputs/ner-from_scratch-{n_str}"
    prompts = [prompt] * (n or 1)
    prompt_dataset = Dataset.from_dict(
        {"idx": list(range(len(prompts))), "prompt": prompts}
    )
    extra_vllm_kwargs = {"limit_mm_per_prompt": {"image": 0, "audio": 0}}
    options = VLLMOfflineRuntimeOptions(
        temperature=parsed_args.temperature,
        top_p=parsed_args.top_p,
        top_k=parsed_args.top_k,
        max_tokens=parsed_args.max_tokens,
    )
    client = VLLMOfflineClient(
        model=parsed_args.model,
        max_model_len=parsed_args.max_model_len,
        gpu_memory_utilization=0.90,
        extra_vllm_kwargs=extra_vllm_kwargs,
    )
    with Annotator(client=client, verbose=True) as anno:
        prepared_dataset, _, _ = anno.prepare_data(
            output_dir=output_dir,
            prompt_template="{prompt}",
            dataset=prompt_dataset,
            prepared_hub_id=parsed_args.hub_id,
            force_data_preparation=parsed_args.force_data_preparation,
        )

        anno.run_annotation(
            output_dir=output_dir,
            prompt_template="{prompt}",
            prepared_dataset=prepared_dataset,
            new_hub_id=parsed_args.hub_id,
            upload_every_n_samples=None,
            options=options,
        )


if __name__ == "__main__":
    main()
