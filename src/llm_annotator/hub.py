"""Restore the Hugging Face Hub backups that an annotation run pushes."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.errors import RevisionNotFoundError

from llm_annotator.annotator import (
    PROGRESS_BACKUP_BRANCH_SUFF,
    PROGRESS_DS_LOCAL_SUBDIR,
    SELECTION_RECORD_FILE,
)
from llm_annotator.logging_utils import get_logger


LOGGER = get_logger("hub")


def _read_idxs(files: list[Path], idx_column: str) -> set[Any]:
    """Collect the sample ids of a set of progress files.

    Args:
        files: The ``.jsonl`` files to read.
        idx_column: Column that holds the sample id.

    Returns:
        Every id that the files hold. A line that is not valid JSON is
        skipped, since an interrupted write can leave a partial last line.

    Raises:
        ValueError: If a row has no ``idx_column``.
    """
    idxs: set[Any] = set()
    for pfin in files:
        with pfin.open("rb") as fhin:
            for raw_line in fhin:
                try:
                    row = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(row, dict):
                    continue
                if idx_column not in row:
                    raise ValueError(
                        f"The progress file '{pfin}' has a row without the"
                        f" index column '{idx_column}', so the backup cannot"
                        " be merged into it. Pass the 'idx_column' that the"
                        " run was annotated with."
                    )
                idxs.add(row[idx_column])
    return idxs


def _write_downloaded_rows(
    *,
    staged: Path,
    progress_dir: Path,
    local_files: list[Path],
    idx_column: str,
) -> tuple[int, int]:
    """Write the downloaded progress files into the progress directory.

    Args:
        staged: Directory that holds the downloaded files.
        progress_dir: Local progress directory to write into.
        local_files: The ``.jsonl`` files that the directory already holds.
        idx_column: Column that holds the sample id.

    Returns:
        The number of files written to, and the number of rows written.
    """
    seen_idxs = _read_idxs(local_files, idx_column) if local_files else None
    num_files = 0
    num_rows = 0
    for staged_file in sorted(staged.glob("*.jsonl")):
        lines: list[str] = []
        with staged_file.open("r", encoding="utf-8") as fhin:
            for line in fhin:
                if not line.strip():
                    continue
                if seen_idxs is not None:
                    row = json.loads(line)
                    if idx_column not in row:
                        raise ValueError(
                            f"The backup file '{staged_file.name}' has a row"
                            f" without the index column '{idx_column}', so"
                            " it cannot be merged into the local progress"
                            " files."
                        )
                    if row[idx_column] in seen_idxs:
                        continue
                    seen_idxs.add(row[idx_column])
                lines.append(line if line.endswith("\n") else line + "\n")

        if not lines:
            continue

        target = progress_dir / staged_file.name
        text = "".join(lines)
        if target.is_file() and (size := target.stat().st_size):
            # An interrupted run can leave a partial last line. Starting a
            # new one keeps that damage on its own line, where the resume
            # logic drops it, instead of gluing two rows together.
            with target.open("rb") as fhin:
                fhin.seek(size - 1)
                if fhin.read(1) != b"\n":
                    text = "\n" + text
        with target.open("a", encoding="utf-8") as fhout:
            fhout.write(text)
        num_files += 1
        num_rows += len(lines)

    return num_files, num_rows


def restore_progress_from_hub(
    *,
    hub_id: str,
    output_dir: str | Path,
    task_prefix: str = "",
    idx_column: str = "idx",
    force: bool = False,
) -> Path:
    """Download a run's Hub progress backup into its output directory.

    The branch ``<task_prefix>progress_backup`` of the dataset repository
    holds the JSONL progress files and the selection record of the run. The
    progress files are written to
    ``<output_dir>/<task_prefix>progress_backup/`` and the record next to
    that directory, which is where ``prepare_data`` and ``run_annotation``
    read them. Run this before the annotation on a machine that has no local
    output directory, so the run resumes instead of annotating every row
    again.

    A progress directory that already holds ``.jsonl`` files is refused
    unless ``force`` is set. With ``force`` the rows are merged per sample
    id and a local row wins, so no finished row is lost. Merging on the id
    rather than on the file length is what makes this safe for a run whose
    files hold different rows under the same name: a local
    ``progress_backup_0.jsonl`` with the ids 50-99 (a server pool finishes
    batches out of order) and a Hub file of the same name with the ids 0-49
    give one file with the ids 0-99, where the longer of the two files would
    drop 50 finished rows. A row that a run deleted on purpose
    (``retry_errors``) comes back when the backup still holds it, so apply
    ``retry_errors`` after the restore.

    Args:
        hub_id: The dataset repository that the run backs up to.
        output_dir: The annotator's output directory. For a pipeline step
            this is ``<output_dir>/<NN>-<name>/annotate/``.
        task_prefix: The task prefix of the run, which names both the branch
            and the local directory. For a pipeline step this is
            ``<name>_``.
        idx_column: Column that holds the sample id. Only read when rows are
            merged into existing progress files.
        force: Whether to merge into a progress directory that already holds
            files.

    Returns:
        The local progress directory that the rows were written to.

    Raises:
        FileExistsError: If the progress directory already holds ``.jsonl``
            files and ``force`` is not set.
        ValueError: If the repository has no backup branch for this task.

    Examples:
        >>> restore_progress_from_hub(  # doctest: +SKIP
        ...     hub_id="me/my-dataset",
        ...     output_dir="outputs/run",
        ...     task_prefix="sentiment_",
        ... )
    """
    branch = f"{task_prefix}{PROGRESS_BACKUP_BRANCH_SUFF}"
    out_dir = Path(output_dir)
    progress_dir = out_dir / f"{task_prefix}{PROGRESS_DS_LOCAL_SUBDIR}"
    record_name = f"{task_prefix}{SELECTION_RECORD_FILE}"

    local_files = (
        sorted(progress_dir.glob("*.jsonl")) if progress_dir.is_dir() else []
    )
    if local_files and not force:
        raise FileExistsError(
            f"'{progress_dir}' already holds {len(local_files)} progress"
            " file(s). Pass force=True ('--force' on the command line) to"
            " merge the backup into them, or restore into an output"
            " directory that has none."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            snapshot_download(
                repo_id=hub_id,
                repo_type="dataset",
                revision=branch,
                allow_patterns=["*.jsonl", record_name],
                local_dir=tmp_dir,
            )
        except RevisionNotFoundError as exc:
            raise ValueError(
                f"The dataset '{hub_id}' has no branch '{branch}', so there"
                " is no progress backup to restore. That branch is deleted"
                " once a run finished, and the final dataset is then on the"
                " 'main' branch of the same repository."
            ) from exc

        staged = Path(tmp_dir)
        progress_dir.mkdir(parents=True, exist_ok=True)
        num_files, num_rows = _write_downloaded_rows(
            staged=staged,
            progress_dir=progress_dir,
            local_files=local_files,
            idx_column=idx_column,
        )

        staged_record = staged / record_name
        local_record = out_dir / record_name
        if not staged_record.is_file():
            LOGGER.info(
                f"The branch '{branch}' holds no '{record_name}'. The"
                " settings of the next run are recorded instead of the ones"
                " that the restored rows were annotated with."
            )
        elif local_record.is_file():
            LOGGER.info(
                f"'{local_record}' already exists, so the record on the"
                " branch is not restored."
            )
        else:
            shutil.copy2(staged_record, local_record)
            LOGGER.info(f"Restored the selection record to '{local_record}'.")

    LOGGER.info(
        f"Restored {num_rows:,} row(s) in {num_files} file(s) from branch"
        f" '{branch}' of '{hub_id}' into '{progress_dir}'."
    )
    return progress_dir
