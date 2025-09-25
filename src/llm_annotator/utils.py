import functools
import hashlib
import sys
import time
from os import cpu_count

import numpy as np
from sklearn.metrics import cohen_kappa_score


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


def tune_bias_term(
    y_true,
    logits,
    *,
    thresholds: np.ndarray | None = None,
    do_tune: bool = True,
    bias_min: float = -0.25,
    bias_max: float = 0.25,
    bias_steps: int = 41,
) -> tuple[float, float]:
    """Tune a scalar logit bias to maximize Quadratic Weighted Kappa (QWK).

    The function searches for an additive bias "b" applied to every logit, then decodes
    predictions and evaluates QWK (via ``cohen_kappa_score(..., weights='quadratic')``)
    between predicted and true labels. Only QWK is optimized and returned.

    Parameters
    ----------
    y_true : array-like (N,)
        Ground-truth integer labels.
    logits : array-like
        Raw logits from the model.
    thresholds : np.ndarray | None
        Explicit bias values to test. If None, construct a linspace.
    do_tune : bool, default True
        If False, skip search and only evaluate bias = 0.0.
    bias_min, bias_max : float
        Range for linspace when ``thresholds`` not provided.
    bias_steps : int
        Number of bias values in the linspace.

    Returns
    -------
    best_qwk : float
        Best QWK score found.
    best_bias : float
        Bias value (logit space) giving ``best_qwk``.
    """

    y_true = np.asarray(y_true).reshape(-1)
    logits = np.asarray(logits)

    if logits.ndim != 2 or logits.shape[1] < 1:
        raise ValueError(f"CORAL mode expects logits shape (N, K-1); got {logits.shape}")

    if thresholds is None:
        thresholds = np.linspace(bias_min, bias_max, bias_steps)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)

    if not do_tune:
        thresholds = np.asarray([0.0], dtype=np.float32)

    best_qwk, best_b = -1.0, 0.0
    K_minus_1 = logits.shape[1]
    for b in thresholds:
        probs = stable_sigmoid(logits + b)  # (N, K-1)
        y_hat = (probs >= 0.5).sum(axis=1).astype(int)  # (N,)
        y_hat = np.clip(y_hat, 0, K_minus_1)
        try:
            qwk = cohen_kappa_score(y_true, y_hat, weights="quadratic")
        except ValueError:
            qwk = -1.0
        if qwk > best_qwk:
            best_qwk, best_b = qwk, float(b)

    return best_qwk, best_b


def stable_sigmoid(x):
    """
    Sigmoid function with clipping to avoid overflow.
    """
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))
