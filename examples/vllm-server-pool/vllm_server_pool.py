"""Annotate a dataset with a pool of vLLM servers, using the library directly.

Every server becomes one worker of a ``VLLMQueueAnnotator``, which keeps a
bounded queue of batches in flight across the whole pool and streams per-sample
results to disk, so an interrupted run can be resumed by re-running the same
command.

This is the Python-API version, useful when the annotation is part of a larger
program. For a run described entirely by a config file -- including multi-step
pipelines where each step has its own model -- see ``pipeline.yaml`` in this
directory and ``slurm/submit_pipeline.sh``, which reach the same annotator
without any Python.
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from pathlib import Path

from datasets import Dataset

from llm_annotator import (
    VLLMOnlineClient,
    VLLMOnlineRuntimeOptions,
    VLLMQueueAnnotator,
)


EXAMPLE_TEXTS = [
    "The notebook auto-saves every few minutes and keeps the previous revision.",
    "The cluster scheduler starts one inference process per allocated node.",
    "A queue-based orchestrator keeps GPU utilization high without blocking.",
    "The model answer is returned as JSON when guided decoding is enabled.",
]

# ``{field}`` is filled in with --prompt-field, so the default template follows
# whichever column the dataset uses.
DEFAULT_PROMPT_TEMPLATE = (
    "Reply with exactly one word: the overall sentiment of this text,"
    " either 'positive' or 'negative'.\n\nText: {{{field}}}\nSentiment:"
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line for this example.

    Args:
        args: Optional argument list; defaults to ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Annotate a dataset over a pool of vLLM servers."
    )
    servers = parser.add_mutually_exclusive_group(required=True)
    servers.add_argument(
        "--base-urls",
        help="Comma-separated vLLM server base URLs.",
    )
    servers.add_argument(
        "--hosts-file",
        type=Path,
        help="File with one vLLM server base URL per line, as collected from"
        " the pool directory by slurm/vllm_annotate.sh.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier served by every vLLM server.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vllm-server-pool"),
        help="Local directory for progress backup and final outputs.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Hub dataset to annotate. Without it, a handful of built-in"
        " example texts is used.",
    )
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument(
        "--prompt-field",
        default="text",
        help="Dataset column referenced by the prompt template.",
    )
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Prompt template; must reference --prompt-field. Defaults to a"
        " simple sentiment prompt over that field.",
    )
    parser.add_argument("--max-num-samples", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of samples sent to a server in one request.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=None,
        help="Batches kept in flight. Defaults to four per server.",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--idx-column",
        default="idx",
        help="Name of the stable identifier column that drives resumption."
        " Must not already exist in the source dataset.",
    )
    parser.add_argument(
        "--task-prefix",
        default="",
        help="Prefix for internal columns and artifact paths, so several tasks"
        " can share one output directory.",
    )
    parser.add_argument(
        "--hub-id",
        default=None,
        help="Optional Hub dataset ID for the progress backup and the final"
        " push.",
    )
    parser.add_argument(
        "--keep-idx-column",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the stable idx column in the final result.",
    )
    parser.add_argument(
        "--wait-for-servers",
        type=float,
        default=0.0,
        help="Seconds to wait for every server to answer /health before"
        " starting (0 disables the check).",
    )
    return parser.parse_args(args)


def resolve_base_urls(parsed: argparse.Namespace) -> list[str]:
    """Collect the vLLM base URLs from the command line or the hosts file.

    Args:
        parsed: Parsed command-line arguments.

    Returns:
        The list of base URLs, in file/argument order.

    Raises:
        ValueError: If no usable URL was given.
    """
    if parsed.hosts_file:
        raw = parsed.hosts_file.read_text(encoding="utf-8").splitlines()
    else:
        raw = parsed.base_urls.split(",")

    urls = [line.strip() for line in raw if line.strip()]
    if not urls:
        raise ValueError("No vLLM server URLs found.")
    return urls


def wait_for_servers(urls: list[str], timeout: float) -> None:
    """Block until every server answers its ``/health`` endpoint.

    Args:
        urls: vLLM base URLs (``.../v1``).
        timeout: Maximum number of seconds to wait per server.

    Raises:
        TimeoutError: If a server is still unreachable after ``timeout``.
    """
    for url in urls:
        health = f"{url.removesuffix('/v1').rstrip('/')}/health"
        deadline = time.monotonic() + timeout
        while True:
            try:
                with urllib.request.urlopen(health, timeout=5) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                pass

            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"vLLM server at '{url}' did not become ready within"
                    f" {timeout:g}s."
                )
            time.sleep(5)


def main(args: list[str] | None = None) -> None:
    """Annotate a dataset over a pool of vLLM servers.

    Args:
        args: Optional argument list; defaults to ``sys.argv``.
    """
    parsed = parse_args(args)
    base_urls = resolve_base_urls(parsed)
    if parsed.wait_for_servers:
        wait_for_servers(base_urls, parsed.wait_for_servers)

    prompt_template = parsed.prompt_template or DEFAULT_PROMPT_TEMPLATE.format(
        field=parsed.prompt_field
    )

    dataset = None
    if parsed.dataset_name is None:
        dataset = Dataset.from_dict({parsed.prompt_field: EXAMPLE_TEXTS})

    print(f"Annotating with {len(base_urls)} vLLM server(s): {base_urls}")
    clients = [
        VLLMOnlineClient(model=parsed.model, base_url=url) for url in base_urls
    ]

    with VLLMQueueAnnotator(
        clients=clients,
        batch_size=parsed.batch_size,
        queue_size=parsed.queue_size,
        verbose=True,
    ) as annotator:
        results = annotator.annotate_dataset(
            output_dir=parsed.output_dir,
            prompt_template=prompt_template,
            dataset=dataset,
            dataset_name=parsed.dataset_name,
            dataset_split=parsed.dataset_split,
            dataset_config=parsed.dataset_config,
            max_num_samples=parsed.max_num_samples,
            idx_column=parsed.idx_column,
            task_prefix=parsed.task_prefix,
            hub_id=parsed.hub_id,
            keep_columns=parsed.prompt_field,
            keep_idx_column=parsed.keep_idx_column,
            options=VLLMOnlineRuntimeOptions(
                max_completion_tokens=parsed.max_completion_tokens,
                temperature=parsed.temperature,
            ),
        )

    print(results)
    print(f"Annotated {len(results):,} samples -> {parsed.output_dir}")


if __name__ == "__main__":
    main()
