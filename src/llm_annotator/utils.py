import functools
import hashlib
import sys
import time
from os import cpu_count


USABLE_CPU_COUNT = cpu_count() - 1


def generate_hash(input_string: str) -> str:
    byte_string = input_string.encode("utf-8")
    hash_object = hashlib.sha256(byte_string)
    hex_hash = hash_object.hexdigest()

    return hex_hash


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


def retry(num_retries: int = 3, sleep_time_s: int = 10) -> callable:
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
