"""Token-counting utilities for RAG benchmark results.

Provides a factory function ``get_tokenizer(name, encoding)`` that returns a
tokenizer object with a ``.name`` attribute and a ``.count(text: str) -> int``
method.

Two implementations are available:

* ``TiktokenTokenizer`` — backed by OpenAI's ``tiktoken`` library (lazy import,
  encoding cached). If ``tiktoken`` is not importable the factory silently
  falls back to ``SimpleTokenizer`` and logs a warning.
* ``SimpleTokenizer`` — a rough estimator: ``math.ceil(len(text) / 4)``.
"""

from __future__ import annotations

import logging
import math
from typing import Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class _Tokenizer(Protocol):
    """Structural contract for tokenizers — duck-typed, not a base class."""

    name: str

    def count(self, text: str) -> int: ...


# ---------------------------------------------------------------------------
# Simple tokenizer
# ---------------------------------------------------------------------------


class SimpleTokenizer:
    """Estimates token count as ``ceil(len(text) / 4)``.

    This is a rough but deterministic approximation that does not depend on any
    external library.  It is used as a fallback when ``tiktoken`` is unavailable.
    """

    name: str = "simple"

    @staticmethod
    def count(text: str) -> int:
        """Return the estimated token count for *text*.

        An empty string always returns 0 (``max(0, …)`` guards against any
        corner-case where ``len('') // 4`` would produce something surprising).
        """
        if not text:
            return 0
        return max(0, math.ceil(len(text) / 4))


# ---------------------------------------------------------------------------
# tiktoken-backed tokenizer
# ---------------------------------------------------------------------------


class TiktokenTokenizer:
    """Tokenizes text using OpenAI's ``tiktoken`` BPE encoding.

    The library is imported on first use (lazy import) and the encoding object
    is cached so repeated calls to ``.count()`` are fast.
    """

    name: str = "tiktoken"

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name: str = encoding_name
        self._encoding: object | None = None  # cached tiktoken.Encoding

    def _get_encoding(self) -> object:
        """Lazy-import tiktoken and return the cached encoding object."""
        if self._encoding is None:
            import tiktoken  # noqa: PLC0415 — lazy import

            self._encoding = tiktoken.get_encoding(self._encoding_name)
        return self._encoding

    def count(self, text: str) -> int:
        """Return the number of tiktoken tokens in *text*.

        An empty string returns 0.
        """
        if not text:
            return 0
        encoding = self._get_encoding()
        tokens: list[int] = encoding.encode(text)
        return len(tokens)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_tokenizer(name: str, encoding: str | None = None) -> _Tokenizer:
    """Return a tokenizer instance for the given *name*.

    Parameters
    ----------
    name:
        ``"tiktoken"`` or ``"simple"`` (case-sensitive).
    encoding:
        The tiktoken encoding name to use (e.g. ``"cl100k_base"``,
        ``"o200k_base"``).  Only meaningful when *name* is ``"tiktoken"``.
        Defaults to ``"cl100k_base"``.

    Returns
    -------
    _Tokenizer
        An object with a ``.name`` attribute and a ``.count(text) -> int``
        method.

    Raises
    ------
    ValueError
        If *name* is not ``"tiktoken"`` or ``"simple"``.

    Notes
    -----
    When *name* is ``"tiktoken"`` but the ``tiktoken`` library cannot be
    imported, a warning is logged and a ``SimpleTokenizer`` is returned
    instead.  This makes ``tiktoken`` an optional dependency — the rest of
    the project works fine without it.
    """
    if name == "simple":
        return SimpleTokenizer()

    if name == "tiktoken":
        try:
            import tiktoken  # noqa: F401, PLC0415
        except ImportError:
            logger.warning(
                "tiktoken is not installed; falling back to simple tokenizer "
                "(math.ceil(len(text) / 4)). Install tiktoken for accurate "
                "BPE token counts."
            )
            return SimpleTokenizer()
        return TiktokenTokenizer(encoding_name=encoding or "cl100k_base")

    raise ValueError(
        f"Unknown tokenizer name: {name!r}. "
        f"Supported names: 'tiktoken', 'simple'."
    )
