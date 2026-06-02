from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_annotator import utils


def test_get_hash_is_stable_and_hex() -> None:
    # Verifies hash outputs are deterministic and valid hexadecimal digests.
    value = utils.get_hash("hello")
    assert value == utils.get_hash("hello")
    assert len(value) == 64
    int(value, 16)


@pytest.mark.parametrize(
    ("num", "expected"),
    [
        (1_000_000_000, "1B"),
        (1_250_000_000, "1.2B"),
        (1_000_000, "1M"),
        (1_234_000, "1.2M"),
        (1_000, "1K"),
        (1_234, "1.2K"),
        (42, "42"),
    ],
)
def test_convert_int_to_annotated_str(num: int, expected: str) -> None:
    # Verifies compact numeric formatting for each magnitude bucket.
    assert utils.convert_int_to_annotated_str(num) == expected


def test_yield_jsonl_robust_handles_keep_columns_dedup_and_corrupt_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Verifies robust reader handles deduplication, field filtering, and corrupt lines.
    p = tmp_path / "data.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"id": 1, "txt": "a", "extra": 1}),
                json.dumps({"id": 2, "txt": "a", "extra": 2}),
                "{bad-json",
                json.dumps({"id": 3, "txt": "b", "extra": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = list(
        utils.yield_jsonl_robust(
            [p],
            keep_columns=["id", "txt"],
            disable_tqdm=True,
            deduplicate_on="txt",
        )
    )

    assert rows == [{"id": 1, "txt": "a"}, {"id": 3, "txt": "b"}]
    out = capsys.readouterr().out
    assert "Skipped 1 corrupt line(s)" in out
    assert "Removed 1 duplicates" in out


def test_count_lines_and_remove_empty_jsonl_files(tmp_path: Path) -> None:
    # Verifies line counting and cleanup of empty jsonl files.
    p_non_empty = tmp_path / "a.jsonl"
    p_non_empty.write_text('{"x": 1}\n{"x": 2}\n', encoding="utf-8")
    p_empty = tmp_path / "b.jsonl"
    p_empty.write_text("", encoding="utf-8")

    assert utils.count_lines(p_non_empty) == 2
    removed = utils.remove_empty_jsonl_files(tmp_path)
    assert removed == [p_empty]
    assert p_non_empty.exists()
    assert not p_empty.exists()


def test_ensure_returns_bool_and_dict() -> None:
    # Verifies return-type guard helpers for bool and dict outputs.
    assert utils.ensure_returns_bool(lambda: True) is True
    assert utils.ensure_returns_dict(lambda: {"k": "v"}) == {"k": "v"}

    with pytest.raises(TypeError, match="should return a bool"):
        utils.ensure_returns_bool(lambda: "yes")

    with pytest.raises(TypeError, match="should return a dict"):
        utils.ensure_returns_dict(lambda: [1, 2])


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
