import hashlib
import json
import re
import sys
from collections.abc import Callable
from functools import lru_cache
from importlib.metadata import version
from os import PathLike
from pathlib import Path
from typing import Any, Generator

from datasets import Dataset
from huggingface_hub import whoami
from tqdm import tqdm

from llm_annotator.logging_utils import get_logger


LOGGER = get_logger("utils")


def get_hash(text: str) -> str:
    """Compute a SHA256 hash for a given text string.

    Args:
        text: The input string to hash.

    Returns:
        A 64-character hexadecimal SHA256 digest.

    Examples:
        >>> len(get_hash("hello"))
        64
        >>> get_hash("hello") == get_hash("hello")
        True
        >>> get_hash("hello") == get_hash("world")
        False
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dataset_signature(dataset: Dataset, num_probe_rows: int = 64) -> str:
    """Compute a content signature of a dataset without reading all of it.

    The signature is a SHA256 hash of the row count, the column names and up to
    ``num_probe_rows`` evenly spaced rows. It depends on the content only, so
    it is the same in every process and for every version of ``datasets``. A
    change in the number of rows always changes it. An edit of a row that is
    not probed does not.

    Args:
        dataset: The dataset to describe.
        num_probe_rows: Maximum number of rows that are hashed.

    Returns:
        A 64-character hexadecimal SHA256 digest.

    Examples:
        >>> from datasets import Dataset
        >>> first = Dataset.from_dict({"text": ["a", "b"]})
        >>> same = Dataset.from_dict({"text": ["a", "b"]})
        >>> longer = Dataset.from_dict({"text": ["a", "b", "c"]})
        >>> dataset_signature(first) == dataset_signature(same)
        True
        >>> dataset_signature(first) == dataset_signature(longer)
        False
    """
    num_rows = len(dataset)
    num_probes = min(num_probe_rows, num_rows)
    probe_idxs = sorted(
        {(i * num_rows) // num_probes for i in range(num_probes)}
    )
    # The Arrow view holds storage values (e.g. image bytes), whose repr is
    # stable. Decoded Python objects can carry a memory address in theirs.
    probes = (
        dataset.select(probe_idxs).with_format("arrow")[:].to_pylist()
        if probe_idxs
        else []
    )
    payload = {
        "num_rows": num_rows,
        "columns": sorted(dataset.column_names),
        "probes": probes,
    }
    return get_hash(json.dumps(payload, sort_keys=True, default=repr))


def convert_int_to_annotated_str(num: int) -> str:
    """Convert an integer to a concise string approximating its magnitude.

    Args:
        num: Non-negative integer to format.

    Returns:
        A compact string representation such as ``"1B"``, ``"1.2M"``, or ``"1.2K"``.

    Examples:
        >>> convert_int_to_annotated_str(1_000_000_000)
        '1B'
        >>> convert_int_to_annotated_str(1_234_567)
        '1.2M'
        >>> convert_int_to_annotated_str(1_234)
        '1.2K'
        >>> convert_int_to_annotated_str(42)
        '42'
    """
    if num >= 1_000_000_000:
        numstr = f"{num / 1_000_000_000:.1f}".rstrip("0").rstrip(
            "."
        )  # remove trailing '.0' if exactly 1 billion
        return f"{numstr}B"
    elif num >= 1_000_000:
        numstr = f"{num / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{numstr}M"
    elif num >= 1_000:
        numstr = f"{num / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{numstr}K"
    else:
        return str(num)


def yield_jsonl_robust(
    pfiles: list[Path | str],
    keep_columns: list[str] | None = None,
    disable_tqdm: bool = False,
    deduplicate_on: str | None = None,
) -> Generator[dict, None, None]:
    """Read a set of ``.jsonl`` files robustly, skipping corrupt lines, and yield one sample at a time.

    Args:
        pfiles: List of ``.jsonl`` file paths to read.
        keep_columns: Columns to retain in each yielded sample. ``None`` keeps all columns.
        disable_tqdm: Whether to suppress the file-level progress bar.
        deduplicate_on: Column name whose value is hashed for deduplication. When
            provided, only the first occurrence of each unique value is yielded.

    Yields:
        One parsed JSON record (``dict``) per non-corrupt line across all files.
    """
    _paths: list[Path] = [Path(pfile) for pfile in pfiles]
    seen = set()
    num_duplicates_removed = 0
    with tqdm(
        total=len(_paths), desc="Reading", unit="file", disable=disable_tqdm
    ) as pbar:
        for pfin in _paths:
            if pfin.stat().st_size == 0:
                continue

            with pfin.open(encoding="utf-8") as fhin:
                num_failures = 0
                while True:
                    try:
                        line = fhin.readline()
                        if not line:
                            break
                        data = json.loads(line)
                        if deduplicate_on:
                            hashed_col = get_hash(data[deduplicate_on])
                            if hashed_col in seen:
                                num_duplicates_removed += 1
                                continue
                            seen.add(hashed_col)

                        if keep_columns:
                            data = {
                                k: v
                                for k, v in data.items()
                                if k in keep_columns
                            }

                        yield data
                    except json.JSONDecodeError:
                        # Handle partial or malformed JSON (incomplete writes)
                        num_failures += 1
                    except EOFError:
                        # Handle unexpected EOF in gzip
                        num_failures += 1
                        break
                if num_failures:
                    print(
                        f"Skipped {num_failures:,} corrupt line(s) in {pfin}"
                    )
            pbar.update(1)

    if deduplicate_on:
        print(f"Removed {num_duplicates_removed:,} duplicates")


def count_lines(fname: str | PathLike) -> int:
    """Count the number of lines in a file.

    Args:
        fname: Path to the file to count lines in.
    Returns:
        The total number of lines in the file.
    """
    with open(fname, "r", encoding="utf-8") as fhin:
        return sum([1 for _ in fhin])


def remove_empty_jsonl_files(pdout: Path) -> list[Path]:
    """Remove any empty .jsonl files in the given directory.

    Args:
        pdout: Output directory path to clean up.

    Returns:
        A list of removed files.
    """
    files_removed = set()
    for pfin in pdout.glob("*.jsonl"):
        if pfin.stat().st_size == 0:
            files_removed.add(pfin)
            pfin.unlink()

    return sorted(files_removed)


@lru_cache(maxsize=None)
def _idx_line_prefix(idx_column: str) -> bytes:
    """Render the bytes that a progress line starts with.

    The result is cached because ``read_jsonl_idx`` needs it once per line of
    a resume scan, where rendering it again is a measurable share of the work.

    Args:
        idx_column: Column that holds the sample id.

    Returns:
        The opening brace, the JSON-encoded column name, a colon and a space.

    Examples:
        >>> _idx_line_prefix("idx")
        b'{"idx": '
    """
    return b"{" + json.dumps(idx_column).encode("utf-8") + b": "


def read_jsonl_idx(
    raw_line: bytes, idx_column: str, *, with_row: bool = False
) -> tuple[Any, dict[str, Any] | None]:
    """Read the sample id out of one line of a progress file.

    The annotator writes ``idx_column`` as the first key of every row, so the
    id can be read from the start of the line instead of from a parse of the
    whole row, which for a long response is most of the work. Three kinds of
    line are parsed in full instead: one that starts with another key, one
    whose id is a string that holds a comma, and one that does not end with
    a closing brace and a newline. The last of those keeps an interrupted
    write detectable, since the writer emits one whole row per line and an
    interrupted write ends mid-row. Pass ``with_row`` when the caller needs
    the other fields too.

    Args:
        raw_line: One line of a ``.jsonl`` progress file, as bytes.
        idx_column: Column that holds the sample id.
        with_row: Whether to parse the whole line and return the row.

    Returns:
        The sample id, and the parsed row when the line was parsed in full.

    Raises:
        json.JSONDecodeError: If the line is not valid JSON.
        UnicodeDecodeError: If the line is not valid UTF-8.
        TypeError: If the line is not a JSON object.
        KeyError: If the row has no ``idx_column``.

    Examples:
        >>> read_jsonl_idx(b'{"idx": 7, "response": "ok"}\\n', "idx")
        (7, None)
        >>> read_jsonl_idx(b'{"response": "ok", "idx": 7}\\n', "idx")
        (7, {'response': 'ok', 'idx': 7})
        >>> read_jsonl_idx(b'{"idx": "a,b", "n": 1}\\n', "idx")
        ('a,b', {'idx': 'a,b', 'n': 1})

        An interrupted write is reported instead of read past:

        >>> read_jsonl_idx(b'{"idx": 7, "respo', "idx")
        Traceback (most recent call last):
        json.decoder.JSONDecodeError: Unterminated string starting at: line 1 column 12 (char 11)
    """
    if not with_row and raw_line.endswith((b"}\n", b"}\r\n")):
        prefix = _idx_line_prefix(idx_column)
        if raw_line.startswith(prefix):
            end = raw_line.find(b",", len(prefix))
            if end != -1:
                try:
                    return json.loads(raw_line[len(prefix) : end]), None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

    row = json.loads(raw_line)
    if not isinstance(row, dict):
        raise TypeError(
            f"a progress row must be a JSON object, got {type(row).__name__}"
        )
    return row[idx_column], row


def drop_jsonl_rows(
    pdout: Path, should_drop: Callable[[dict[str, Any]], bool]
) -> list[dict[str, Any]]:
    """Remove the rows that match a predicate from every .jsonl file in a directory.

    A file that loses rows is written to a temporary file that then replaces
    it, so a crash leaves either the old or the new file. A line that is not
    valid JSON is kept as it is.

    Args:
        pdout: Directory that holds the ``*.jsonl`` files.
        should_drop: Called with each parsed row. ``True`` removes the row.

    Returns:
        The removed rows.

    Examples:
        >>> import tempfile
        >>> pdout = Path(tempfile.mkdtemp())
        >>> _ = (pdout / "rows.jsonl").write_text('{"idx": 0}\\n{"idx": 1}\\n')
        >>> drop_jsonl_rows(pdout, lambda row: row["idx"] == 1)
        [{'idx': 1}]
        >>> (pdout / "rows.jsonl").read_text()
        '{"idx": 0}\\n'
    """
    dropped: list[dict[str, Any]] = []
    for pfin in sorted(pdout.glob("*.jsonl")):
        kept_lines: list[bytes] = []
        num_dropped_before = len(dropped)
        with pfin.open("rb") as fhin:
            for raw_line in fhin:
                try:
                    row = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    row = None
                if isinstance(row, dict) and should_drop(row):
                    dropped.append(row)
                else:
                    kept_lines.append(raw_line)

        if len(dropped) > num_dropped_before:
            pftmp = pfin.with_suffix(".jsonl.tmp")
            pftmp.write_bytes(b"".join(kept_lines))
            pftmp.replace(pfin)

    return dropped


def ensure_returns_bool(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> bool:
    """Ensure that a callable returns a boolean value.

    Args:
        func: Callable to invoke.
        *args: Positional arguments forwarded to ``func``.
        **kwargs: Keyword arguments forwarded to ``func``.

    Returns:
        The boolean result returned by ``func``.

    Raises:
        TypeError: If ``func`` does not return a boolean.
    """
    result = func(*args, **kwargs)
    if not isinstance(result, bool):
        raise TypeError(
            f"{func.__name__} should return a bool, got {type(result).__name__}"
        )
    return result


def ensure_returns_dict(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> dict[str, Any]:
    """Ensure that a callable returns a dictionary.

    Args:
        func: Callable to invoke.
        *args: Positional arguments forwarded to ``func``.
        **kwargs: Keyword arguments forwarded to ``func``.

    Returns:
        The dictionary result returned by ``func``.

    Raises:
        TypeError: If ``func`` does not return a dictionary.
    """
    result = func(*args, **kwargs)
    if not isinstance(result, dict):
        raise TypeError(
            f"{func.__name__} should return a dict, got {type(result).__name__}"
        )
    return result


def get_lib_versions() -> dict[str, str]:
    """Get the versions of key dependencies."""

    ver = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
    }

    libraries = ("transformers", "torch", "vllm", "openai", "anthropic")

    for lib in libraries:
        try:
            ver[lib] = version(lib)
        except Exception:
            ver[lib] = "not installed"

    try:
        # May fail if llm-annotator is not installed, which can happen eg in containers
        # when src/ is just added to PYTHONPATH without a full pip install.
        llm_annotator_version = version("llm_annotator")
    except Exception:
        llm_annotator_version = "unknown"

    ver["llm_annotator"] = llm_annotator_version

    return ver


def get_hf_username() -> str | None:
    """Get the Hugging Face username of the current user, if logged in. Otherwise, return None.

    Returns:
        The Hugging Face username, or None if not logged in.
    """
    try:
        whowasi = whoami()
    except Exception:
        return None

    if whowasi and "name" in whowasi and whowasi["type"] == "user":
        return str(whowasi["name"])
    return None


_PLACEHOLDER_RE = re.compile(r"\{[^}]+\}")


def extract_prompt_prefix(prompt: str) -> str:
    """Extract the prefix of a prompt up to the first ``{placeholder}``, or the entire prompt if none exists.

    Can return an empty string when the prompt starts with a ``{placeholder}``.
    This is expected when using ``generate_dataset`` with fully variable prompts.

    Args:
        prompt: The full prompt string, optionally containing ``{field}`` placeholders.

    Returns:
        The substring before the first ``{placeholder}``, or the entire prompt when
        no placeholder is present.

    Examples:
        >>> extract_prompt_prefix("Classify: {text}")
        'Classify: '
        >>> extract_prompt_prefix("{text} is the input")
        ''
        >>> extract_prompt_prefix("No placeholders here")
        'No placeholders here'
    """
    return re.split(_PLACEHOLDER_RE, prompt, maxsplit=1)[0]


def add_schema_additional_properties_false(schema: Any) -> Any:
    """Recursively set ``additionalProperties: false`` on all object schemas.

    Claude requires this on every object type in the schema; without it the
    API returns a 400 error.

    Args:
        schema: A JSON-schema dict (or any nested value).

    Returns:
        A new schema dict with ``additionalProperties`` set to ``False`` on
        every sub-schema whose ``type`` is ``"object"``.
    """
    if not isinstance(schema, dict):
        return schema
    schema = {
        k: add_schema_additional_properties_false(v) for k, v in schema.items()
    }
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
    return schema


def is_in_range(
    value: int | float,
    min_value: int | float | None,
    max_value: int | float | None,
) -> bool:
    """Check if a numeric value falls within an optional range (inclusive). Utility function
    that models can use for validation.

    Args:
        value: The numeric value to check.
        min_value: The minimum allowed value (inclusive), or None for no minimum.
        max_value: The maximum allowed value (inclusive), or None for no maximum.

    Returns:
        True if the value is within the range, False otherwise.
    """
    if min_value is not None and value < min_value:
        return False
    if max_value is not None and value > max_value:
        return False
    return True


def is_length(
    text: str, min_length: int | None, max_length: int | None
) -> bool:
    """Check if the length of a text string falls within an optional range. Utility function
    that models can use for validation.

    Args:
        text: The text string to check.
        min_length: The minimum allowed length (inclusive), or None for no minimum.
        max_length: The maximum allowed length (inclusive), or None for no maximum.
    Returns:
        True if the text length is within the range, False otherwise.
    """
    length = len(text)
    return is_in_range(length, min_length, max_length)


__all__ = [
    "add_schema_additional_properties_false",
    "convert_int_to_annotated_str",
    "is_in_range",
    "count_lines",
    "dataset_signature",
    "drop_jsonl_rows",
    "ensure_returns_bool",
    "ensure_returns_dict",
    "extract_prompt_prefix",
    "get_hash",
    "get_hf_username",
    "get_lib_versions",
    "remove_empty_jsonl_files",
    "yield_jsonl_robust",
]
