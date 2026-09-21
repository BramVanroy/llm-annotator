from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset

from llm_annotator import utils


def test_get_hash_is_stable_and_hex() -> None:
    # Verifies hash outputs are deterministic and valid hexadecimal digests.
    value = utils.get_hash("hello")
    assert value == utils.get_hash("hello")
    assert len(value) == 64
    int(value, 16)


def test_remove_empty_jsonl_files(tmp_path: Path) -> None:
    # Verifies cleanup of empty jsonl files.
    p_non_empty = tmp_path / "a.jsonl"
    p_non_empty.write_text('{"x": 1}\n{"x": 2}\n', encoding="utf-8")
    p_empty = tmp_path / "b.jsonl"
    p_empty.write_text("", encoding="utf-8")

    removed = utils.remove_empty_jsonl_files(tmp_path)
    assert removed == [p_empty]
    assert p_non_empty.exists()
    assert not p_empty.exists()


def test_drop_jsonl_rows_drops_matches_and_keeps_untouched_files(
    tmp_path: Path,
) -> None:
    # Verifies matching rows are dropped from one file while a file without
    # any match is left byte-identical, and a broken trailing line survives.
    matching = tmp_path / "matching.jsonl"
    matching_bytes = (
        json.dumps({"idx": 0, "error": None}).encode()
        + b"\n"
        + json.dumps({"idx": 1, "error": "boom"}).encode()
        + b"\n"
        + b'{"idx": 2, "error": "half-writ'
    )
    matching.write_bytes(matching_bytes)

    untouched = tmp_path / "untouched.jsonl"
    untouched_bytes = json.dumps({"idx": 3, "error": None}).encode() + b"\n"
    untouched.write_bytes(untouched_bytes)
    untouched_inode_before = untouched.stat().st_ino

    dropped = utils.drop_jsonl_rows(
        tmp_path, lambda row: row.get("error") is not None
    )

    assert dropped == [{"idx": 1, "error": "boom"}]
    assert matching.read_bytes() == (
        json.dumps({"idx": 0, "error": None}).encode()
        + b"\n"
        + b'{"idx": 2, "error": "half-writ'
    )
    # An untouched file is never rewritten, not merely rewritten unchanged.
    assert untouched.read_bytes() == untouched_bytes
    assert untouched.stat().st_ino == untouched_inode_before


def test_ensure_returns_bool_and_dict() -> None:
    # Verifies return-type guard helpers for bool and dict outputs.
    sample = {"k": "v"}
    assert utils.ensure_returns_bool(lambda s: bool(s), sample) is True
    assert utils.ensure_returns_dict(lambda s: s, sample) == {"k": "v"}

    with pytest.raises(TypeError, match="should return a bool"):
        utils.ensure_returns_bool(lambda s: "yes", sample)

    with pytest.raises(TypeError, match="should return a dict"):
        utils.ensure_returns_dict(lambda s: [1, 2], sample)


def test_get_lib_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Verifies dependency version collection and python version formatting.
    monkeypatch.setattr(utils, "version", lambda name: f"{name}-v")
    versions = utils.get_lib_versions()
    assert versions["llm_annotator"] == "llm_annotator-v"
    assert versions["vllm"] == "vllm-v"
    assert versions["torch"] == "torch-v"
    assert versions["transformers"] == "transformers-v"
    assert len(versions["python"].split(".")) == 3


def test_get_hf_username(monkeypatch: pytest.MonkeyPatch) -> None:
    # Verifies HF username extraction for user, org, and unauthenticated states.
    monkeypatch.setattr(
        utils, "whoami", lambda: {"name": "alice", "type": "user"}
    )
    assert utils.get_hf_username() == "alice"

    monkeypatch.setattr(
        utils, "whoami", lambda: {"name": "org", "type": "org"}
    )
    assert utils.get_hf_username() is None

    def _raise() -> None:
        raise RuntimeError("not logged in")

    monkeypatch.setattr(utils, "whoami", _raise)
    assert utils.get_hf_username() is None


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Classify: {text}", "Classify: "),
        ("{text} as input", ""),
        ("No placeholders", "No placeholders"),
        ("First {a} then {b}", "First "),
    ],
)
def test_extract_prompt_prefix(prompt: str, expected: str) -> None:
    # Verifies prompt prefix extraction before the first template placeholder.
    assert utils.extract_prompt_prefix(prompt) == expected


def test_add_schema_additional_properties_false() -> None:
    # Verifies object schemas are closed recursively while preserving existing settings.
    schema = {
        "type": "object",
        "properties": {
            "child": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
            },
            "locked": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer"},
                },
                "additionalProperties": True,
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                    },
                },
            },
        },
    }

    updated = utils.add_schema_additional_properties_false(schema)

    assert updated["additionalProperties"] is False
    assert updated["properties"]["child"]["additionalProperties"] is False
    assert (
        updated["properties"]["items"]["items"]["additionalProperties"]
        is False
    )
    assert updated["properties"]["locked"]["additionalProperties"] is True


def test_dataset_signature_is_stable_for_equal_content() -> None:
    # Two datasets built independently from the same content must agree.
    first = Dataset.from_dict({"text": [f"row {i}" for i in range(10)]})
    second = Dataset.from_dict({"text": [f"row {i}" for i in range(10)]})
    assert utils.dataset_signature(first) == utils.dataset_signature(second)


def test_dataset_signature_survives_a_disk_round_trip(tmp_path: Path) -> None:
    # The same content must hash the same after save_to_disk/load_from_disk,
    # not just for a freshly built in-memory dataset.
    dataset = Dataset.from_dict({"text": [f"row {i}" for i in range(10)]})
    before = utils.dataset_signature(dataset)

    path = tmp_path / "ds"
    dataset.save_to_disk(str(path))
    reloaded = Dataset.load_from_disk(str(path))

    assert utils.dataset_signature(reloaded) == before


def test_dataset_signature_differs_for_a_different_row_count() -> None:
    fewer = Dataset.from_dict({"text": [f"row {i}" for i in range(10)]})
    more = Dataset.from_dict({"text": [f"row {i}" for i in range(11)]})
    assert utils.dataset_signature(fewer) != utils.dataset_signature(more)


def test_dataset_signature_differs_for_a_different_column_name() -> None:
    original = Dataset.from_dict({"text": ["a", "b"]})
    renamed = original.rename_column("text", "other")
    assert utils.dataset_signature(original) != utils.dataset_signature(
        renamed
    )


def test_dataset_signature_differs_for_a_changed_probed_row() -> None:
    # With few rows every row is probed, so editing any one of them changes
    # the signature.
    original = Dataset.from_dict({"text": ["a", "b", "c"]})
    changed = Dataset.from_dict({"text": ["a", "X", "c"]})
    assert utils.dataset_signature(original) != utils.dataset_signature(
        changed
    )


def test_dataset_signature_works_on_an_empty_dataset() -> None:
    empty = Dataset.from_dict({"text": []})
    assert utils.dataset_signature(empty) == utils.dataset_signature(empty)


def test_dataset_signature_is_stable_for_a_bytes_column() -> None:
    # The Arrow-view probe must not choke on a column whose values are raw
    # bytes rather than decoded Python objects.
    first = Dataset.from_dict({"data": [b"abc", b"def"]})
    second = Dataset.from_dict({"data": [b"abc", b"def"]})
    assert utils.dataset_signature(first) == utils.dataset_signature(second)

    other = Dataset.from_dict({"data": [b"abc", b"xyz"]})
    assert utils.dataset_signature(first) != utils.dataset_signature(other)


def test_get_lib_versions_marks_a_library_not_installed_when_version_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A dependency whose version() call raises is reported as not
    # installed, while the other dependencies still resolve normally.
    def _version(name: str) -> str:
        if name == "torch":
            raise ModuleNotFoundError(name)
        return f"{name}-v"

    monkeypatch.setattr(utils, "version", _version)
    versions = utils.get_lib_versions()

    assert versions["torch"] == "not installed"
    assert versions["transformers"] == "transformers-v"


def test_get_lib_versions_falls_back_to_unknown_for_its_own_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # llm_annotator's own version() call can fail, e.g. when only src/ is on
    # PYTHONPATH without an installed package; that case reports "unknown".
    def _version(name: str) -> str:
        if name == "llm_annotator":
            raise ModuleNotFoundError(name)
        return f"{name}-v"

    monkeypatch.setattr(utils, "version", _version)
    versions = utils.get_lib_versions()

    assert versions["llm_annotator"] == "unknown"
