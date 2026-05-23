from pathlib import Path

from llm_annotator import (
    Annotator,
    VLLMOfflineClient,
    VLLMRuntimeOptions,
)


CURR_DIR = Path(__file__).parent


def main(args=None):
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
    args = parser.parse_args(args)
    if args.max_num_samples == -1:
        args.max_num_samples = None

    prompt = CURR_DIR.joinpath("prompt_template.md").read_text(
        encoding="utf-8"
    )
    n = args.max_num_samples
    n_str = str(n) if n is not None else "full"
    output_dir = args.output_dir or f"outputs/ner-from_scratch-{n_str}"
    extra_vllm_kwargs = {"limit_mm_per_prompt": {"image": 0, "audio": 0}}
    options = VLLMRuntimeOptions(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    client = VLLMOfflineClient(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.95,
        extra_vllm_kwargs=extra_vllm_kwargs,
    )
    with Annotator(client=client, verbose=True) as anno:
        anno.generate_dataset(
            output_dir=output_dir,
            prompts=prompt,
            max_num_samples=n,
            upload_every_n_samples=None,
            options=options,
        )


if __name__ == "__main__":
    main()
