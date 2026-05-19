"""Pure-math statistical primitives for rag-bench.

No I/O, no dependencies on the database or server layer.
Exports:
    wilson_ci(k, n, alpha=0.05)          → (ci_low, ci_high)
    normal_ci(mean, std, n, alpha=0.05)  → (ci_low, ci_high)
    pearson_correlation_matrix(rows, columns)  → ndarray
"""
from __future__ import annotations

from typing import Union

import numpy as np
from scipy.stats import norm, t


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_positive_int(value: float, name: str, *, gt_zero: bool = True) -> None:
    """Raise ``ValueError`` if *value* is not a positive integer."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a numeric value, got {type(value).__name__}")
    if gt_zero and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    if not gt_zero and value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


ArrayLike = Union[list, tuple, np.ndarray]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wilson_ci(
    k: Union[int, float],
    n: Union[int, float],
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Wilson score interval (95 % by default) for a proportion.

    Parameters
    ----------
    k : number of successes (0 <= k <= n)
    n : total trials (n > 0)
    alpha : significance level (two-sided)

    Returns
    -------
    tuple[float, float] : (ci_low, ci_high)
    """
    _check_positive_int(n, "n", gt_zero=True)
    _check_positive_int(k, "k", gt_zero=False)
    if k > n:
        raise ValueError(f"k ({k}) must be <= n ({n})")
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    p_hat = k / n
    z = norm.ppf(1 - alpha / 2)
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denom
    half_width = z * np.sqrt(
        p_hat * (1 - p_hat) / n + z * z / (4.0 * n * n)
    ) / denom
    return float(centre - half_width), float(centre + half_width)


def normal_ci(
    mean: Union[int, float],
    std: Union[int, float],
    n: Union[int, float],
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Normal/t confidence interval for a mean.

    Uses Student's *t* distribution (df = n-1) for small samples
    and the normal distribution for n >= 1.

    Parameters
    ----------
    mean : sample mean
    std : sample standard deviation
    n : sample size (n > 0)
    alpha : significance level (two-sided)

    Returns
    -------
    tuple[float, float] : (ci_low, ci_high)
    """
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    if std < 0:
        raise ValueError(f"std must be >= 0, got {std}")

    df = n - 1
    if df <= 0:
        # Single observation: fall back to normal
        z = norm.ppf(1 - alpha / 2)
    else:
        z = t.ppf(1 - alpha / 2, df=df)
    se = std / np.sqrt(n)
    margin = z * se
    return float(mean - margin), float(mean + margin)


def pearson_correlation_matrix(
    *data: ArrayLike,
) -> np.ndarray:
    """Pearson correlation matrix from column vectors.

    Parameters
    ----------
    *data : each argument is a sequence of numbers (same length).

    Returns
    -------
    np.ndarray : square correlation matrix, shape (k, k).

    Raises
    ------
    ValueError : if arguments differ in length or are all-constant.
    """
    arrays = [np.asarray(d, dtype=np.float64) for d in data]
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError("All data arrays must have the same length")
    if not lengths:
        raise ValueError("Data arrays must not be empty")

    if len(arrays[0]) == 0:
        raise ValueError("Data arrays must not be empty")

    stack = np.column_stack(arrays)  # shape (n, k)
    return np.corrcoef(stack, rowvar=False)
