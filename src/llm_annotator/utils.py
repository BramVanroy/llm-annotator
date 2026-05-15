import functools
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable
from importlib.metadata import version
from os import PathLike
from pathlib import Path
from typing import Generator

from huggingface_hub import whoami
from tqdm import tqdm


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


def retry(num_retries: int = 3, sleep_time_s: int = 1) -> Callable:
    """Return a decorator that retries a callable on failure with exponential back-off.

    Useful for network operations such as uploading data to Hugging Face Hub.
    The wait time doubles after each failed attempt.

    Args:
        num_retries: Maximum number of retry attempts before re-raising the exception.
        sleep_time_s: Initial wait time in seconds before the first retry.

    Returns:
        A decorator that wraps a callable with automatic retry logic.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retries_left = num_retries
            current_sleep_time = sleep_time_s
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if retries_left <= 0:
                        print(
                            f"Function {func.__name__} failed after {num_retries} retries.",
                            file=sys.stderr,
                        )
                        raise exc

                    print(
                        f"Function {func.__name__} failed with {exc}. Retrying in {current_sleep_time}s... ({retries_left} retries left)",
                        file=sys.stderr,
                    )
                    time.sleep(current_sleep_time)
                    retries_left -= 1
                    current_sleep_time *= 2

        return wrapper

    return decorator


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


def ensure_returns_bool(func, *args, **kwargs):
    """Ensure that the given function returns a boolean value. If not, raise a TypeError."""
    result = func(*args, **kwargs)
    if not isinstance(result, bool):
        raise TypeError(
            f"{func.__name__} should return a bool, got {type(result).__name__}"
        )
    return result


def ensure_returns_dict(func, *args, **kwargs):
    """Ensure that the given function returns a dict value. If not, raise a TypeError."""
    result = func(*args, **kwargs)
    if not isinstance(result, dict):
        raise TypeError(
            f"{func.__name__} should return a dict, got {type(result).__name__}"
        )
    return result


def get_lib_versions() -> dict[str, str]:
    """Get the versions of key dependencies."""

    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "llm_annotator": version("llm_annotator"),
        "vllm": version("vllm"),
        "torch": version("torch"),
        "transformers": version("transformers"),
    }


def get_hf_username() -> str | None:
    """Get the Hugging Face username of the current user, if logged in. Otherwise, return None.

    Returns:
        The Hugging Face username, or None if not logged in.
    Raises:
        LocalTokenNotFoundError: If no local token is found.
    """
    whowasi = whoami()
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
