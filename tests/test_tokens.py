"""Tests for the tokenizer module (rag_bench/tokens.py)."""

import logging
import math
import sys

import pytest


# ---------------------------------------------------------------------------
# VAL-TOK-001: Tokenizer factory exposes tiktoken and simple
# ---------------------------------------------------------------------------

class TestGetTokenizerFactory:
    """Factory returns tokenizers with .name and .count(text) -> int."""

    def test_get_tokenizer_tiktoken_returns_object_with_name_and_count(self):
        from rag_bench.tokens import get_tokenizer

        tok = get_tokenizer("tiktoken")
        assert hasattr(tok, "name")
        assert tok.name == "tiktoken"
        assert hasattr(tok, "count")
        assert callable(tok.count)

    def test_get_tokenizer_simple_returns_object_with_name_and_count(self):
        from rag_bench.tokens import get_tokenizer

        tok = get_tokenizer("simple")
        assert hasattr(tok, "name")
        assert tok.name == "simple"
        assert hasattr(tok, "count")
        assert callable(tok.count)


# ---------------------------------------------------------------------------
# VAL-TOK-002: tiktoken tokenizer returns actual tiktoken counts
# ---------------------------------------------------------------------------

class TestTiktokenCountAccuracy:
    """When tiktoken is installed, counts must match tiktoken.get_encoding."""

    def test_tiktoken_count_matches_tiktoken_encode_length(self):
        import tiktoken

        from rag_bench.tokens import get_tokenizer

        text = "def hello_world() -> None: pass"
        tok = get_tokenizer("tiktoken", "cl100k_base")
        expected = len(tiktoken.get_encoding("cl100k_base").encode(text))
        assert tok.count(text) == expected

    def test_tiktoken_count_returns_zero_for_empty(self):
        from rag_bench.tokens import get_tokenizer

        tok = get_tokenizer("tiktoken")
        assert tok.count("") == 0

    def test_tiktoken_count_positive_for_non_empty(self):
        from rag_bench.tokens import get_tokenizer

        tok = get_tokenizer("tiktoken")
        assert tok.count("hello world") > 0

    def test_tiktoken_default_encoding_is_cl100k_base(self):
        import tiktoken

        from rag_bench.tokens import get_tokenizer

        tok = get_tokenizer("tiktoken")
        text = "some code here"
        expected = len(tiktoken.get_encoding("cl100k_base").encode(text))
        assert tok.count(text) == expected

    def test_tiktoken_custom_encoding(self):
        import tiktoken

        from rag_bench.tokens import get_tokenizer

        text = "print('hello')"
        tok = get_tokenizer("tiktoken", "o200k_base")
        expected = len(tiktoken.get_encoding("o200k_base").encode(text))
        assert tok.count(text) == expected


# ---------------------------------------------------------------------------
# VAL-TOK-003: simple tokenizer estimates length / 4
# ---------------------------------------------------------------------------

class TestSimpleTokenizerCount:
    """Simple tokenizer: max(0, ceil(len(text) / 4))."""

    def test_simple_count_matches_ceil_len_div_4(self):
        from rag_bench.tokens import get_tokenizer

        tok = get_tokenizer("simple")
        for text in ["a", "ab", "abc", "abcd", "abcde", "hello world"]:
            expected = max(0, math.ceil(len(text) / 4))
            assert tok.count(text) == expected, f"failed for {text!r}: got {tok.count(text)}, expected {expected}"

    def test_simple_count_zero_for_empty(self):
        from rag_bench.tokens import get_tokenizer

        tok = get_tokenizer("simple")
        assert tok.count("") == 0

    def test_simple_count_positive_for_non_empty(self):
        from rag_bench.tokens import get_tokenizer

        tok = get_tokenizer("simple")
        assert tok.count("x") > 0


# ---------------------------------------------------------------------------
# VAL-TOK-004: graceful fallback when tiktoken not installed
# ---------------------------------------------------------------------------

class TestTiktokenFallback:
    """When tiktoken cannot be imported, get_tokenizer('tiktoken') falls back."""

    def test_fallback_to_simple_when_tiktoken_import_fails(self, monkeypatch, caplog):
        """Monkeypatch tiktoken import to raise ImportError; verify fallback."""
        # Remove tiktoken from sys.modules so the import is re-attempted
        monkeypatch.setitem(sys.modules, "tiktoken", None)

        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken" or name.startswith("tiktoken."):
                raise ImportError("tiktoken not available (mocked)")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        from rag_bench.tokens import get_tokenizer

        with caplog.at_level(logging.WARNING):
            tok = get_tokenizer("tiktoken")

        assert tok.name == "simple"
        assert tok.count("hello") == max(0, math.ceil(len("hello") / 4))

        # A WARNING should have been logged about the fallback
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("tiktoken" in w.lower() or "simple" in w.lower() for w in warnings), (
            f"Expected fallback warning, got: {warnings}"
        )


# ---------------------------------------------------------------------------
# VAL-TOK-005: unknown tokenizer name raises ValueError
# ---------------------------------------------------------------------------

class TestUnknownTokenizer:
    """get_tokenizer('nonsense') raises ValueError."""

    def test_unknown_name_raises_value_error(self):
        from rag_bench.tokens import get_tokenizer

        with pytest.raises(ValueError, match="nonsense"):
            get_tokenizer("nonsense")

    def test_unknown_name_message_mentions_name(self):
        from rag_bench.tokens import get_tokenizer

        with pytest.raises(ValueError) as exc_info:
            get_tokenizer("unknown-tokenizer")
        assert "unknown-tokenizer" in str(exc_info.value)


# ---------------------------------------------------------------------------
# VAL-TOK-006: tokenizer counts are non-negative and stable
# ---------------------------------------------------------------------------

class TestCountStability:
    """Repeated calls must return identical values. Non-empty text > 0."""

    def test_count_stable_for_tiktoken(self):
        from rag_bench.tokens import get_tokenizer

        tok = get_tokenizer("tiktoken")
        text = "def foo(bar: int) -> str: return str(bar)"
        first = tok.count(text)
        for _ in range(10):
            assert tok.count(text) == first

    def test_count_stable_for_simple(self):
        from rag_bench.tokens import get_tokenizer

        tok = get_tokenizer("simple")
        text = "some arbitrary text for testing"
        first = tok.count(text)
        for _ in range(10):
            assert tok.count(text) == first

    def test_count_empty_returns_zero_for_both(self):
        from rag_bench.tokens import get_tokenizer

        assert get_tokenizer("tiktoken").count("") == 0
        assert get_tokenizer("simple").count("") == 0

    def test_count_non_empty_positive_for_both(self):
        from rag_bench.tokens import get_tokenizer

        assert get_tokenizer("tiktoken").count("hello") > 0
        assert get_tokenizer("simple").count("hello") > 0
