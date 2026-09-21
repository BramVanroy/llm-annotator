"""Restore a run's Hub progress backup into its local output directory.

Use this on a machine that has no local progress files, for example after a
scratch directory was purged or when the run moves to another cluster::

    python scripts/restore_progress_from_hub.py \
        --hub-id user/my-dataset --output-dir outputs/run

All of the logic lives in :func:`llm_annotator.hub.restore_progress_from_hub`,
which is where to look (and where to add tests) when changing behaviour.
"""

import argparse
from pathlib import Path

from llm_annotator.hub import restore_progress_from_hub
from llm_annotator.logging_utils import configure_logging


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download the JSONL progress backup and the selection record of"
            " an annotation run from the Hugging Face Hub, so that a rerun"
            " resumes instead of annotating every row again."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hub-id",
        required=True,
        help="Dataset repository that the run backs up to.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "The annotator's output directory. For a pipeline step this is"
            " <output_dir>/<NN>-<name>/annotate/."
        ),
    )
    parser.add_argument(
        "--task-prefix",
        default="",
        help=(
            "Task prefix of the run, which names both the Hub branch and the"
            " local progress directory. For a pipeline step this is <name>_."
        ),
    )
    parser.add_argument(
        "--idx-column",
        default="idx",
        help=(
            "Column that holds the sample id. Only read when rows are merged"
            " into existing progress files."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Merge into a progress directory that already holds files. Rows"
            " are merged per sample id and a local row wins."
        ),
    )

    configure_logging()
    restore_progress_from_hub(**vars(parser.parse_args()))
