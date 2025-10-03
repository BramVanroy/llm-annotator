import functools
import hashlib
import json
import sys
import time
from os import PathLike
from pathlib import Path

from tqdm import tqdm


def get_hash(text: str) -> str:
    """Compute a SHA256 hash for a given text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def convert_int_to_str(num: int) -> str:
    """Convert an integer to a concise string approximating the `num`.
    E.g. 1_000_000 -> '1M', 1_234_567 -> '1.23M', 1_234 -> '1.23K'
    """
    if num >= 1_000_000_000:
        numstr = f"{num / 1_000_000_000:.1f}".rstrip("0").rstrip(".")  # remove trailing '.0' if exactly 1 billion
        return f"{numstr}B"
    elif num >= 1_000_000:
        numstr = f"{num / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{numstr}M"
    elif num >= 1_000:
        numstr = f"{num / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{numstr}K"
    else:
        return str(num)


def retry(num_retries: int = 3, sleep_time_s: int = 1) -> callable:
    """
    A decorator to automatically retry a function if it fails. Useful when we are uploading data.

    Args:
        num_retries (int): The maximum number of times to retry the function.
        sleep_time_s (int): The initial time in seconds to wait before retrying.
        This time will double after each failed attempt.
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
                        print(f"Function {func.__name__} failed after {num_retries} retries.", file=sys.stderr)
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
):
    """
    Given a set of .jsonl.gz files, this function reads them in a robust way, skipping incomplete lines,
    and yielding one sample at a time (parse-able JSON line).

    :param pfiles: A list of .jsonl.gz files
    :param keep_columns: A list of columns to keep in the output. If not given, all columns are kept.
    :param disable_tqdm: Whether to disable the progress bar
    :param deduplicate_on: Column name to use for deduplication (will be hashed)
    :return: A generator yielding the contents of the files
    """
    pfiles = [Path(pfile) for pfile in pfiles]
    seen = set()
    num_duplicates_removed = 0
    with tqdm(total=len(pfiles), desc="Reading", unit="file", disable=disable_tqdm) as pbar:
        for pfin in pfiles:
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
                            data = {k: v for k, v in data.items() if k in keep_columns}

                        yield data
                    except json.JSONDecodeError:
                        # Handle partial or malformed JSON (incomplete writes)
                        num_failures += 1
                    except EOFError:
                        # Handle unexpected EOF in gzip
                        num_failures += 1
                        break
                if num_failures:
                    print(f"Skipped {num_failures:,} corrupt line(s) in {pfin}")
            pbar.update(1)

    if deduplicate_on:
        print(f"Removed {num_duplicates_removed:,} duplicates")


def count_lines(fname: str | PathLike) -> int:
    """Count the number of lines in a file."""
    with open(fname, "r", encoding="utf-8") as fhin:
        return sum([1 for _ in fhin])


def remove_empty_jsonl_files(pdout: Path) -> list[Path]:
    """Remove any empty .jsonl files in the given directory.

    Args:
        pdout: Output directory path to clean up.

    Returns:
        A list of removed files.
    """
    files_removed = []
    for pfin in pdout.glob("*.jsonl"):
        if pfin.stat().st_size == 0:
            files_removed.append(pfin)
            pfin.unlink()

    return files_removed
