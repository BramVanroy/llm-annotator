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


def test_read_jsonl_idx_fast_path_returns_id_only_when_id_is_first() -> None:
    # The fast path reads the id straight from the start of the line and
    # does not parse the rest of the row.
    line = json.dumps({"idx": 7, "response": "ok"}).encode() + b"\n"
    assert utils.read_jsonl_idx(line, "idx") == (7, None)


def test_read_jsonl_idx_fallback_returns_the_row_when_id_is_not_first() -> (
    None
):
    # A row whose id is not the first key falls back to a full parse and
    # returns that row instead of None.
    line = json.dumps({"response": "ok", "idx": 7}).encode() + b"\n"
    assert utils.read_jsonl_idx(line, "idx") == (
        7,
        {"response": "ok", "idx": 7},
    )


def test_read_jsonl_idx_fallback_handles_a_comma_inside_the_id_string() -> (
    None
):
    # A string id that holds a comma cannot be located by scanning for the
    # first comma, so this case must fall back to a full parse.
    line = json.dumps({"idx": "a,b", "n": 1}).encode() + b"\n"
    assert utils.read_jsonl_idx(line, "idx") == (
        "a,b",
        {"idx": "a,b", "n": 1},
    )


def test_read_jsonl_idx_fast_path_handles_a_crlf_line_ending() -> None:
    line = json.dumps({"idx": 3, "response": "ok"}).encode() + b"\r\n"
    assert utils.read_jsonl_idx(line, "idx") == (3, None)


def test_read_jsonl_idx_truncated_line_raises_json_decode_error() -> None:
    # A line with no closing brace and newline is an interrupted write, even
    # when it starts with the id key, and must be reported, not read past.
    line = b'{"idx": 7, "respo'
    with pytest.raises(json.JSONDecodeError):
        utils.read_jsonl_idx(line, "idx")


def test_read_jsonl_idx_array_line_raises_type_error() -> None:
    line = b"[1, 2]\n"
    with pytest.raises(TypeError):
        utils.read_jsonl_idx(line, "idx")


def test_read_jsonl_idx_missing_id_column_raises_key_error() -> None:
    line = json.dumps({"response": "ok"}).encode() + b"\n"
    with pytest.raises(KeyError):
        utils.read_jsonl_idx(line, "idx")


def test_read_jsonl_idx_with_row_true_always_returns_the_parsed_row() -> None:
    # with_row=True skips the fast path even when the id is first, since the
    # caller needs the rest of the row.
    line = json.dumps({"idx": 7, "response": "ok"}).encode() + b"\n"
    idx, row = utils.read_jsonl_idx(line, "idx", with_row=True)
    assert idx == 7
    assert row == {"idx": 7, "response": "ok"}


def test_read_jsonl_idx_handles_an_id_column_name_needing_escaping() -> None:
    # The id column name is JSON-encoded before it is matched against the
    # start of the line, so a name that needs escaping still hits the fast
    # path.
    idx_column = 'my"idx'
    line = json.dumps({idx_column: 5, "response": "ok"}).encode() + b"\n"
    assert utils.read_jsonl_idx(line, idx_column) == (5, None)
