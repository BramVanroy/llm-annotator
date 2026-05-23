from pathlib import Path

from llm_annotator import Annotator, VLLMOfflineClient, VLLMRuntimeOptions
from llm_annotator.utils import get_hf_username


CURR_DIR = Path(__file__).parent


def main(args=None):
    import argparse

    # TODO: add speculative_config, which should be a string but which we should then json.loads
    # TODO: add reasoner_parser to vlm offline client
    # TODO: check how reasoning/thinking should be implemented

    parser = argparse.ArgumentParser(
        description="Translate finemath to Dutch."
    )
    parser.add_argument(
        "--model", default="RedHatAI/gemma-3-27b-it-FP8-dynamic"
    )
    parser.add_argument("--max-model-len", type=int, default=72_000)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument(
        "--max-num-samples",
        type=int,
        default=10,
        help="Number of samples to process. Use -1 for the full dataset.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=36_000)
    parser.add_argument(
        "--dataset", default="BramVanroy/finemath-4plus-seqlen36k"
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--hub-id",
        default=None,
        help="HF Hub dataset ID template; use {hf_user} and {n} as placeholders.",
    )
    parser.add_argument(
        "--speculative-config",
        default=None,
        help="JSON string for speculative decoding config.",
    )
    args = parser.parse_args(args)
    if args.max_num_samples == -1:
        args.max_num_samples = None

    hf_user = get_hf_username()
    prompt_template = CURR_DIR.joinpath("prompt_template.md").read_text(
        encoding="utf-8"
    )

    n = args.max_num_samples
    n_str = str(n) if n is not None else "full"
    hub_id = args.hub_id
    if hub_id is not None:
        hub_id = hub_id.format(hf_user=hf_user, n=n_str)
    elif hf_user:
        hub_id = f"{hf_user}/finemath-dutch-{n_str}"

    options = VLLMRuntimeOptions(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    client = VLLMOfflineClient(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.95,
    )
    with Annotator(client=client, verbose=True) as anno:
        anno.annotate_dataset(
            output_dir=f"outputs/finemath-dutch-{n}",
            prompt_template=prompt_template,
            dataset_name=args.dataset,
            dataset_split=args.dataset_split,
            new_hub_id=hub_id,
            max_num_samples=n,
            keep_columns=True,
            upload_every_n_samples=None,
            options=options,
        )


if __name__ == "__main__":
    main()
