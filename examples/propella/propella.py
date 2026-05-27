from llm_annotator import (
    Annotator,
    VLLMOfflineClient,
    VLLMOfflineRuntimeOptions,
)
from llm_annotator.external.propella.propella import (
    ANNOTATOR_USER_PROMPT,
    annotator_system_prompt,
    get_annotation_response_schema,
)
from llm_annotator.logging_utils import get_logger


logger = get_logger(__name__)

MODEL_ID_BY_SIZE = {
    "0.6b": "ellamind/propella-1-0.6b",
    "1.7b": "ellamind/propella-1-1.7b",
    "4b": "ellamind/propella-1-4b",
}


def main(args=None):
    """Run Propella-based annotation with the canonical Propella prompt."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Annotate dataset text with propella-1 using the canonical full "
            "Propella prompt and schema."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Input dataset name/path.",
    )
    parser.add_argument(
        "--dataset_config",
        help="Input dataset configuration name.",
    )
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--text-column",
        required=True,
        help="Dataset column containing input text.",
    )

    parser.add_argument(
        "--model-size",
        choices=("0.6b", "1.7b", "4b"),
        default="4b",
        help="Propella model size alias.",
    )
    parser.add_argument(
        "--use-fp8",
        action="store_true",
        default=False,
        help="Enable fp8 quantization for supported model sizes (1.7b, 4b).",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where annotated output is written.",
    )
    parser.add_argument(
        "--hub-id",
        default=None,
        help="Optional HF Hub dataset ID.",
    )

    parser.add_argument("--max-model-len", type=int, default=65_536)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--max-num-samples",
        type=int,
        default=-1,
        help="Number of samples to process. Use -1 for full split.",
    )

    args = parser.parse_args(args)

    if args.max_num_samples == -1:
        args.max_num_samples = None

    model_id = MODEL_ID_BY_SIZE[args.model_size]

    if args.use_fp8 and "propella-1-0.6b" in model_id:
        raise ValueError(
            "--use-fp8 is only supported for model sizes 1.7b and 4b."
        )

    output_schema = get_annotation_response_schema(
        one_sentence_description_max_length=150
    )
    prompt_template = ANNOTATOR_USER_PROMPT
    system_message = annotator_system_prompt

    hub_id = args.hub_id

    options = VLLMOfflineRuntimeOptions(max_tokens=args.max_tokens)

    quantization = "fp8" if args.use_fp8 else None
    extra_vllm_kwargs = {"kv_cache_dtype": "fp8"} if args.use_fp8 else None

    client = VLLMOfflineClient(
        model=model_id,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        batch_size=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=quantization,
        extra_vllm_kwargs=extra_vllm_kwargs,
    )

    if args.max_num_samples is None:
        upload_every_n_samples = 2_500
    else:
        upload_every_n_samples = max(1, args.max_num_samples // 4)

    with Annotator(
        client=client,
        verbose=True,
        batch_size=args.max_num_seqs,
    ) as anno:
        anno.annotate_dataset(
            output_dir=args.output_dir,
            prompt_template=prompt_template,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            new_hub_id=hub_id,
            max_num_samples=args.max_num_samples,
            keep_columns=True,
            upload_every_n_samples=upload_every_n_samples,
            options=options,
            output_schema=output_schema,
            system_message=system_message,
            prompt_field_swapper={"content": args.text_column},
        )


if __name__ == "__main__":
    main()
