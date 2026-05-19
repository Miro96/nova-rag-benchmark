"""Pure-math statistical primitives for rag-bench.

No I/O, no dependencies on the database or server layer.
Exports:
    wilson_ci(k, n, alpha=0.05)          → (ci_low, ci_high)
    normal_ci(mean, std, n, alpha=0.05)  → (ci_low, ci_high)
    pearson_correlation_matrix(rows, columns)  → ndarray
    cohens_d(x, y)                        → float
    cliffs_delta(x, y)                    → float
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


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cohens_d(x: ArrayLike, y: ArrayLike) -> float:
    """Pooled Cohen's d for two independent samples.

    d = (mean(x) - mean(y)) / s_pooled
    s_pooled = sqrt(((n_x - 1) * var(x) + (n_y - 1) * var(y)) / (n_x + n_y - 2))

    Parameters
    ----------
    x, y : sequences of numbers (same length expected for paired context).

    Returns
    -------
    float : Cohen's d. Positive means x > y.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n_x, n_y = len(x_arr), len(y_arr)

    if n_x < 2 and n_y < 2:
        return 0.0

    mean_x = np.mean(x_arr)
    mean_y = np.mean(y_arr)
    var_x = np.var(x_arr, ddof=1) if n_x > 1 else 0.0
    var_y = np.var(y_arr, ddof=1) if n_y > 1 else 0.0

    if n_x + n_y <= 2:
        return 0.0

    s_pooled = np.sqrt(
        ((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2)
    )
    if s_pooled == 0:
        return 0.0
    return float((mean_x - mean_y) / s_pooled)


def cliffs_delta(x: ArrayLike, y: ArrayLike) -> float:
    """Cliff's delta — a non-parametric effect-size measure.

    delta = (#(x_i > y_j) - #(x_i < y_j)) / (n_x * n_y)

    Returns a value in [-1, 1].  Positive means x tends to be larger
    than y.  A value of 1.0 means every x_i > every y_j.

    Parameters
    ----------
    x, y : sequences of comparable values.

    Returns
    -------
    float : Cliff's delta.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n_x, n_y = len(x_arr), len(y_arr)
    if n_x == 0 or n_y == 0:
        return 0.0

    greater = 0
    less = 0
    # Vectorised comparison
    for xi in x_arr:
        greater += int(np.sum(xi > y_arr))
        less += int(np.sum(xi < y_arr))

    denom = n_x * n_y * 1.0
    return float((greater - less) / denom)
