import hashlib
import json
import re
import sys
from collections.abc import Callable
from importlib.metadata import version
from os import PathLike
from pathlib import Path
from typing import Any, Generator

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
    "ensure_returns_bool",
    "ensure_returns_dict",
    "extract_prompt_prefix",
    "get_hash",
    "get_hf_username",
    "get_lib_versions",
    "remove_empty_jsonl_files",
    "yield_jsonl_robust",
]
