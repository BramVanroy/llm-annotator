from __future__ import annotations

import ast
import importlib.util
import json
import logging
import shutil
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from datasets import Dataset
from huggingface_hub.errors import RevisionNotFoundError

from llm_annotator.annotator import Annotator, SelectionRecord
from llm_annotator.clients.base import (
    Client,
    Provider,
    ProviderRuntimeOptions,
    Response,
)
from llm_annotator.hub import restore_progress_from_hub


RESTORE_SCRIPT = (
    Path(__file__).parent.parent / "scripts" / "restore_progress_from_hub.py"
)


class EchoClient(Client[ProviderRuntimeOptions]):
    """Answers every prompt with its own text and records what it saw."""

    provider_type = Provider.OPENAI

    def __init__(self) -> None:
        super().__init__(model="dummy", on_error="raise")
        self.seen_prompts: list[str] = []

    def _process_response(self, response: str) -> Response:
        return Response(
            text=response, provider=self.provider_type, model=self.model
        )

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> Response:
        _ = options
        _ = gen_kwargs
        self.seen_prompts.append(messages[-1]["content"])
        return Response(
            text=messages[-1]["content"],
            stop_reason="stop",
            provider=self.provider_type,
            model=self.model,
            num_output_tokens=3,
        )

    def batch_generate(
        self,
        *,
        messages: list[list[dict[str, str]]],
        options: ProviderRuntimeOptions | None = None,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> list[Response]:
        return [
            self.generate(messages=msg, options=options, gen_kwargs=gen_kwargs)
            for msg in messages
        ]

    def _handle_stop_reason(
        self, *, stop_reason: str | None, num_output_tokens: int | None
    ) -> None:
        _ = stop_reason
        _ = num_output_tokens


def _write_jsonl(path: Path, idxs: list[int], response: str = "ok") -> None:
    """Write one progress row per id to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"idx": idx, "response": response}) + "\n"
            for idx in idxs
        ),
        encoding="utf-8",
    )


@pytest.fixture
def fake_download(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Path], list[dict[str, Any]]]:
    """Make ``snapshot_download`` copy a local directory into ``local_dir``.

    Returns a factory that takes the directory to serve and returns the list
    of calls that the annotator made.
    """

    def _serve(branch_dir: Path) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def _fake(**kwargs: Any) -> str:
            calls.append(kwargs)
            local_dir = Path(kwargs["local_dir"])
            patterns = kwargs["allow_patterns"]
            for pfin in sorted(branch_dir.iterdir()):
                if any(fnmatch(pfin.name, pat) for pat in patterns):
                    shutil.copy2(pfin, local_dir / pfin.name)
            return str(local_dir)

        monkeypatch.setattr("llm_annotator.hub.snapshot_download", _fake)
        return calls

    return _serve


def test_restore_writes_progress_files_and_record(
    tmp_path: Path, fake_download: Callable[[Path], list[dict[str, Any]]]
) -> None:
    # Verifies the branch's files land in the progress directory and the
    # record next to it, and that only those patterns are requested.
    branch_dir = tmp_path / "branch"
    _write_jsonl(branch_dir / "progress_backup_0.jsonl", [0, 1])
    _write_jsonl(branch_dir / "progress_backup_1.jsonl", [2])
    (branch_dir / "selection.json").write_text(
        json.dumps(
            {
                "max_num_samples": 3,
                "source_rows": 3,
                "selected_rows": 3,
                "components": {"prompt_template": "abc"},
            }
        ),
        encoding="utf-8",
    )
    calls = fake_download(branch_dir)

    progress_dir = restore_progress_from_hub(
        hub_id="me/ds", output_dir=tmp_path / "out"
    )

    assert progress_dir == tmp_path / "out" / "progress_backup"
    assert sorted(p.name for p in progress_dir.glob("*.jsonl")) == [
        "progress_backup_0.jsonl",
        "progress_backup_1.jsonl",
    ]
    record = SelectionRecord.read(tmp_path / "out")
    assert record is not None
    assert record.components == {"prompt_template": "abc"}
    assert calls[0]["revision"] == "progress_backup"
    assert calls[0]["allow_patterns"] == ["*.jsonl", "selection.json"]
    assert calls[0]["repo_type"] == "dataset"


def test_restore_uses_the_task_prefix_for_branch_and_directory(
    tmp_path: Path, fake_download: Callable[[Path], list[dict[str, Any]]]
) -> None:
    # Verifies a prefixed task reads its own branch and writes its own paths.
    branch_dir = tmp_path / "branch"
    _write_jsonl(branch_dir / "judge_progress_backup_0.jsonl", [0])
    (branch_dir / "judge_selection.json").write_text(
        json.dumps(
            {
                "max_num_samples": None,
                "source_rows": 1,
                "selected_rows": 1,
                "components": {"prompt_template": "x"},
            }
        ),
        encoding="utf-8",
    )
    calls = fake_download(branch_dir)

    progress_dir = restore_progress_from_hub(
        hub_id="me/ds", output_dir=tmp_path / "out", task_prefix="judge_"
    )

    assert progress_dir == tmp_path / "out" / "judge_progress_backup"
    assert (progress_dir / "judge_progress_backup_0.jsonl").is_file()
    assert (tmp_path / "out" / "judge_selection.json").is_file()
    assert calls[0]["revision"] == "judge_progress_backup"
    assert calls[0]["allow_patterns"] == ["*.jsonl", "judge_selection.json"]


def test_restore_refuses_an_existing_progress_directory(
    tmp_path: Path, fake_download: Callable[[Path], list[dict[str, Any]]]
) -> None:
    # Verifies local progress files are never touched without force.
    branch_dir = tmp_path / "branch"
    _write_jsonl(branch_dir / "progress_backup_0.jsonl", [0, 1])
    calls = fake_download(branch_dir)

    local = tmp_path / "out" / "progress_backup"
    _write_jsonl(local / "progress_backup_0.jsonl", [5], response="local")

    with pytest.raises(FileExistsError, match="already holds 1 progress"):
        restore_progress_from_hub(hub_id="me/ds", output_dir=tmp_path / "out")

    assert not calls
    assert json.loads(
        (local / "progress_backup_0.jsonl").read_text().splitlines()[0]
    ) == {"idx": 5, "response": "local"}


def test_restore_force_merges_rows_by_idx(
    tmp_path: Path, fake_download: Callable[[Path], list[dict[str, Any]]]
) -> None:
    # Verifies the union rule: a local row wins over the Hub row with the
    # same id, and a shorter local file keeps its own rows.
    branch_dir = tmp_path / "branch"
    _write_jsonl(branch_dir / "progress_backup_0.jsonl", [0, 1, 2], "hub")
    fake_download(branch_dir)

    local = tmp_path / "out" / "progress_backup"
    _write_jsonl(local / "progress_backup_0.jsonl", [2, 9], response="local")

    restore_progress_from_hub(
        hub_id="me/ds", output_dir=tmp_path / "out", force=True
    )

    rows = [
        json.loads(line)
        for line in (local / "progress_backup_0.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {row["idx"]: row["response"] for row in rows} == {
        0: "hub",
        1: "hub",
        2: "local",
        9: "local",
    }
    assert len(rows) == 4


def test_restore_force_appends_after_a_partial_last_line(
    tmp_path: Path, fake_download: Callable[[Path], list[dict[str, Any]]]
) -> None:
    # Verifies an interrupted write keeps its own line instead of being
    # glued to the first restored row.
    branch_dir = tmp_path / "branch"
    _write_jsonl(branch_dir / "progress_backup_0.jsonl", [0], "hub")
    fake_download(branch_dir)

    local = tmp_path / "out" / "progress_backup"
    local.mkdir(parents=True)
    (local / "progress_backup_0.jsonl").write_text(
        '{"idx": 7, "response": "local"}\n{"idx": 8, "res',
        encoding="utf-8",
    )

    restore_progress_from_hub(
        hub_id="me/ds", output_dir=tmp_path / "out", force=True
    )

    lines = (local / "progress_backup_0.jsonl").read_text().splitlines()
    assert lines[-1] == json.dumps({"idx": 0, "response": "hub"})
    assert lines[1] == '{"idx": 8, "res'


def test_restore_force_needs_the_idx_column(
    tmp_path: Path, fake_download: Callable[[Path], list[dict[str, Any]]]
) -> None:
    # Verifies a merge against rows without the id column is refused.
    branch_dir = tmp_path / "branch"
    _write_jsonl(branch_dir / "progress_backup_0.jsonl", [0])
    fake_download(branch_dir)

    local = tmp_path / "out" / "progress_backup"
    _write_jsonl(local / "progress_backup_0.jsonl", [1])

    with pytest.raises(ValueError, match="index column 'sample_id'"):
        restore_progress_from_hub(
            hub_id="me/ds",
            output_dir=tmp_path / "out",
            idx_column="sample_id",
            force=True,
        )


def test_restore_force_rejects_a_backup_without_the_idx_column(
    tmp_path: Path, fake_download: Callable[[Path], list[dict[str, Any]]]
) -> None:
    # Verifies a downloaded row without the id column names its file.
    branch_dir = tmp_path / "branch"
    branch_dir.mkdir()
    (branch_dir / "progress_backup_0.jsonl").write_text(
        json.dumps({"response": "hub"}) + "\n", encoding="utf-8"
    )
    fake_download(branch_dir)

    local = tmp_path / "out" / "progress_backup"
    _write_jsonl(local / "progress_backup_0.jsonl", [1])

    with pytest.raises(ValueError, match="progress_backup_0.jsonl"):
        restore_progress_from_hub(
            hub_id="me/ds", output_dir=tmp_path / "out", force=True
        )


def test_restore_missing_branch_names_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Verifies the documented message for a branch that a finished run
    # deleted, rather than the raw Hub error.
    response = httpx.Response(
        404, request=httpx.Request("GET", "https://huggingface.co")
    )

    def _missing(**kwargs: Any) -> str:
        _ = kwargs
        raise RevisionNotFoundError("404", response=response)

    monkeypatch.setattr("llm_annotator.hub.snapshot_download", _missing)

    with pytest.raises(ValueError) as excinfo:
        restore_progress_from_hub(
            hub_id="me/ds", output_dir=tmp_path / "out", task_prefix="judge_"
        )

    message = str(excinfo.value)
    assert "me/ds" in message
    assert "judge_progress_backup" in message
    assert "'main'" in message
    assert isinstance(excinfo.value.__cause__, RevisionNotFoundError)


def test_restore_reports_a_branch_without_a_record(
    tmp_path: Path,
    fake_download: Callable[[Path], list[dict[str, Any]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Verifies a backup pushed before the first record is restored anyway.
    branch_dir = tmp_path / "branch"
    _write_jsonl(branch_dir / "progress_backup_0.jsonl", [0, 1])
    fake_download(branch_dir)

    with caplog.at_level(logging.INFO, logger="llm_annotator.hub"):
        restore_progress_from_hub(hub_id="me/ds", output_dir=tmp_path / "out")

    messages = [rec.message for rec in caplog.records]
    assert any("holds no 'selection.json'" in msg for msg in messages)
    assert any("Restored 2 row(s) in 1 file(s)" in msg for msg in messages)
    assert not SelectionRecord.path(tmp_path / "out").is_file()


def test_restore_keeps_an_existing_local_record(
    tmp_path: Path,
    fake_download: Callable[[Path], list[dict[str, Any]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Verifies the record of the local output directory wins.
    branch_dir = tmp_path / "branch"
    _write_jsonl(branch_dir / "progress_backup_0.jsonl", [0])
    (branch_dir / "selection.json").write_text(
        json.dumps(
            {
                "max_num_samples": None,
                "source_rows": 1,
                "selected_rows": 1,
                "components": {"prompt_template": "hub"},
            }
        ),
        encoding="utf-8",
    )
    fake_download(branch_dir)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    SelectionRecord(
        max_num_samples=None,
        source_rows=1,
        selected_rows=1,
        components={"prompt_template": "local"},
    ).write(out_dir)

    with caplog.at_level(logging.INFO, logger="llm_annotator.hub"):
        restore_progress_from_hub(hub_id="me/ds", output_dir=out_dir)

    record = SelectionRecord.read(out_dir)
    assert record is not None
    assert record.components == {"prompt_template": "local"}
    assert any("is not restored" in rec.message for rec in caplog.records)


def test_run_annotation_skips_exactly_the_restored_idxs(
    tmp_path: Path,
    fake_download: Callable[[Path], list[dict[str, Any]]],
) -> None:
    # Verifies a restored backup is what the resume reads: only the rows
    # that the backup does not hold reach the client.
    branch_dir = tmp_path / "branch"
    _write_jsonl(branch_dir / "progress_backup_0.jsonl", [0, 2])
    fake_download(branch_dir)

    out_dir = tmp_path / "out"
    restore_progress_from_hub(hub_id="me/ds", output_dir=out_dir)

    prepared_ds = Dataset.from_dict(
        {
            "idx": [0, 1, 2, 3],
            "messages": [
                [{"role": "user", "content": f"Q: {i}"}] for i in range(4)
            ],
        }
    )
    client = EchoClient()

    result = Annotator(client=client).run_annotation(
        output_dir=out_dir,
        prompt_template="Q: {text}",
        prepared_dataset=prepared_ds,
        upload_every_n_samples=0,
        keep_idx_column=True,
    )

    assert client.seen_prompts == ["Q: 1", "Q: 3"]
    assert sorted(result["idx"]) == [0, 1, 2, 3]


def test_restore_script_parses_and_imports() -> None:
    # Verifies the CLI keeps valid syntax and a side-effect-free module
    # level, the same check that tests/test_examples.py applies to examples.
    source = RESTORE_SCRIPT.read_text(encoding="utf-8")
    ast.parse(source, filename=str(RESTORE_SCRIPT))

    spec = importlib.util.spec_from_file_location(
        "_script_restore_progress_from_hub", RESTORE_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.integration
def test_restore_round_trip_on_the_hub(
    tmp_path: Path, test_remote_dataset_name: str
) -> None:
    # Verifies a real push and restore against a temporary dataset repo.
    from huggingface_hub import delete_repo

    source_dir = tmp_path / "source"
    _write_jsonl(
        source_dir / "progress_backup" / "progress_backup_0.jsonl", [0, 1]
    )
    SelectionRecord(
        max_num_samples=2,
        source_rows=2,
        selected_rows=2,
        components={"prompt_template": "abc"},
    ).write(source_dir)

    annotator = Annotator(client=EchoClient())
    try:
        annotator.push_progress_to_hub(
            source_dir / "progress_backup", hub_id=test_remote_dataset_name
        )
        progress_dir = restore_progress_from_hub(
            hub_id=test_remote_dataset_name, output_dir=tmp_path / "restored"
        )
    finally:
        delete_repo(
            test_remote_dataset_name, repo_type="dataset", missing_ok=True
        )

    rows = [
        json.loads(line)
        for line in (progress_dir / "progress_backup_0.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["idx"] for row in rows] == [0, 1]
    restored = SelectionRecord.read(tmp_path / "restored")
    assert restored is not None
    assert restored.components == {"prompt_template": "abc"}


def test_restored_record_keeps_the_checks_of_the_first_machine(
    tmp_path: Path,
    fake_download: Callable[[Path], list[dict[str, Any]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Verifies the point of shipping the record with the backup: a second
    # machine reuses the restored rows without a warning and refuses an
    # edited prompt, exactly as the first one would.
    source = Dataset.from_dict({"text": ["a", "b"]})
    annotator = Annotator(client=EchoClient())

    machine_a = tmp_path / "a"
    annotator.prepare_data(
        output_dir=machine_a, prompt_template="Q: {text}", dataset=source
    )
    _write_jsonl(machine_a / "progress_backup" / "progress_0.jsonl", [0])

    branch_dir = tmp_path / "branch"
    branch_dir.mkdir()
    shutil.copy2(
        machine_a / "progress_backup" / "progress_0.jsonl", branch_dir
    )
    shutil.copy2(machine_a / "selection.json", branch_dir)
    fake_download(branch_dir)

    machine_b = tmp_path / "b"
    restore_progress_from_hub(hub_id="me/ds", output_dir=machine_b)

    with caplog.at_level(logging.WARNING, logger="llm_annotator.annotator"):
        annotator.prepare_data(
            output_dir=machine_b, prompt_template="Q: {text}", dataset=source
        )
    assert not [
        rec
        for rec in caplog.records
        if rec.levelno >= logging.WARNING
        and rec.name == "llm_annotator.annotator"
    ]

    with pytest.raises(ValueError, match="cannot be reused"):
        annotator.prepare_data(
            output_dir=machine_b, prompt_template="A: {text}", dataset=source
        )


def test_restore_force_skips_blank_and_unusable_lines(
    tmp_path: Path, fake_download: Callable[[Path], list[dict[str, Any]]]
) -> None:
    # Verifies a blank line in the backup, a local line that holds no row,
    # and a backup file whose rows are all local already.
    branch_dir = tmp_path / "branch"
    branch_dir.mkdir()
    (branch_dir / "progress_backup_0.jsonl").write_text(
        json.dumps({"idx": 0, "response": "hub"})
        + "\n\n"
        + json.dumps({"idx": 1, "response": "hub"})
        + "\n",
        encoding="utf-8",
    )
    (branch_dir / "progress_backup_1.jsonl").write_text(
        json.dumps({"idx": 7, "response": "hub"}) + "\n", encoding="utf-8"
    )
    fake_download(branch_dir)

    local = tmp_path / "out" / "progress_backup"
    local.mkdir(parents=True)
    (local / "progress_backup_1.jsonl").write_text(
        '"not a row"\n' + json.dumps({"idx": 7, "response": "local"}) + "\n",
        encoding="utf-8",
    )

    restore_progress_from_hub(
        hub_id="me/ds", output_dir=tmp_path / "out", force=True
    )

    assert (local / "progress_backup_0.jsonl").read_text().splitlines() == [
        json.dumps({"idx": 0, "response": "hub"}),
        json.dumps({"idx": 1, "response": "hub"}),
    ]
    assert (local / "progress_backup_1.jsonl").read_text().splitlines() == [
        '"not a row"',
        json.dumps({"idx": 7, "response": "local"}),
    ]
