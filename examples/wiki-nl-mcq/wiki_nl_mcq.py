import json
from pathlib import Path

from llm_annotator import Annotator, VLLMOfflineClient, VLLMRuntimeOptions
from llm_annotator.utils import get_hf_username


CURR_DIR = Path(__file__).parent
OUTPUT_ROOT_DIR = CURR_DIR.parent.parent.joinpath("outputs")


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Dutch Wikipedia MCQ dataset."
    )
    parser.add_argument(
        "--model", default="RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8"
    )
    parser.add_argument("--max-model-len", type=int, default=48_000)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument(
        "--max-num-samples",
        type=int,
        default=None,
        help="Number of samples to process. Use -1 for the full dataset.",
    )
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--max-tokens", type=int, default=12_000)
    parser.add_argument(
        "--dataset", default="BramVanroy/finewiki-nl-30-to-24k-tokens"
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_ROOT_DIR / "wiki-nl-mcq")
    )
    parser.add_argument(
        "--hub-id", default=None, help="HF Hub dataset ID to push to."
    )
    args = parser.parse_args(args)
    if args.max_num_samples == -1:
        args.max_num_samples = None

    hf_user = get_hf_username()
    prompt_template = CURR_DIR.joinpath("prompt_template.md").read_text(
        encoding="utf-8"
    )
    system_message = CURR_DIR.joinpath("system_prompt.md").read_text(
        encoding="utf-8"
    )
    json_schema = json.loads(
        CURR_DIR.joinpath("output.schema.json").read_text(encoding="utf-8")
    )

    hub_id = args.hub_id or (f"{hf_user}/wiki-nl-mcq" if hf_user else None)
    options = VLLMRuntimeOptions(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    extra_vllm_kwargs = {
        "load_format": "mistral",
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "limit_mm_per_prompt": {"image": 0},
    }
    client = VLLMOfflineClient(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.95,
        extra_vllm_kwargs=extra_vllm_kwargs,
    )
    with Annotator(client=client, verbose=True) as anno:
        anno.annotate_dataset(
            output_dir=args.output_dir,
            prompt_template=prompt_template,
            dataset_name=args.dataset,
            dataset_split=args.dataset_split,
            new_hub_id=hub_id,
            max_num_samples=args.max_num_samples,
            keep_columns=["text", "title", "url"],
            upload_every_n_samples=None,
            options=options,
            system_message=system_message,
            sort_by_length=True,
            output_schema=json_schema,
        )


if __name__ == "__main__":
    main()
