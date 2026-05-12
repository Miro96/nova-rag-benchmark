"""BM25 lexical retrieval baseline.

Chunks code by function/class boundaries via tree-sitter, tokenizes with a
whitespace tokenizer, and scores with BM25Okapi from the rank_bm25 library.
No embeddings — pure lexical matching to serve as a classical IR baseline.

Uses the InProcessClient contract: ``list_tools()`` and ``call_tool(name, **params)``.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# File extensions that are considered "code" for the baseline.
_CODE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
    ".scala", ".cs", ".m", ".mm",
}

# ---------------------------------------------------------------------------
# Tree-sitter language loaders (lazy)
# ---------------------------------------------------------------------------

_LANG_CACHE: dict[str, Any] = {}


def _get_language(suffix: str) -> Any | None:
    """Return a tree-sitter Language for *suffix*, or None."""
    from tree_sitter import Language

    if suffix in _LANG_CACHE:
        return _LANG_CACHE[suffix]

    lang = None
    try:
        if suffix == ".py":
            import tree_sitter_python as tsp
            lang = Language(tsp.language())
        elif suffix in (".js", ".jsx", ".mjs", ".cjs"):
            import tree_sitter_javascript as tsj
            lang = Language(tsj.language())
        elif suffix in (".ts", ".tsx"):
            import tree_sitter_typescript as tst
            lang = Language(tst.language_typescript())
        elif suffix == ".java":
            import tree_sitter_java as tsjv
            lang = Language(tsjv.language())
        elif suffix in (".c", ".h"):
            import tree_sitter_c as tsc
            lang = Language(tsc.language())
        elif suffix in (".cpp", ".hpp"):
            import tree_sitter_cpp as tscpp
            lang = Language(tscpp.language())
        elif suffix == ".go":
            import tree_sitter_go as tsgo
            lang = Language(tsgo.language())
        elif suffix == ".rs":
            import tree_sitter_rust as tsrust
            lang = Language(tsrust.language())
        elif suffix == ".rb":
            import tree_sitter_ruby as tsrb
            lang = Language(tsrb.language())
        elif suffix == ".php":
            import tree_sitter_php as tsphp
            lang = Language(tsphp.language())
        elif suffix == ".swift":
            import tree_sitter_swift as tssw
            lang = Language(tssw.language())
        elif suffix == ".kt":
            import tree_sitter_kotlin as tskt
            lang = Language(tskt.language())
        elif suffix == ".scala":
            import tree_sitter_scala as tssc
            lang = Language(tssc.language())
        elif suffix in (".cs",):
            import tree_sitter_c_sharp as tscs
            lang = Language(tscs.language())
    except Exception:
        logger.debug("Failed to load tree-sitter language for %s", suffix)

    _LANG_CACHE[suffix] = lang
    return lang


# ---------------------------------------------------------------------------
# Regex-based chunking (fallback for unsupported languages)
# ---------------------------------------------------------------------------

_PY_DEF_PATTERN = re.compile(
    r"^(\s*)(?:(?:async\s+)?def\s+\w+|class\s+\w+)",
    re.MULTILINE,
)
_JS_DEF_PATTERN = re.compile(
    r"^(\s*)(?:(?:async\s+)?function\s+\w+|class\s+\w+|"
    r"(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)",
    re.MULTILINE,
)


def _chunk_by_regex(
    content: str, file_path: str, lang_pattern: re.Pattern,
) -> list[dict[str, Any]]:
    """Split *content* at function/class definition boundaries (regex fallback)."""
    lines = content.split("\n")
    matches = list(lang_pattern.finditer(content))

    if not matches:
        text = content.strip()
        if not text:
            return []
        return [{
            "file_path": file_path,
            "content": text,
            "start_line": 1,
            "symbol": Path(file_path).stem,
        }]

    chunks: list[dict[str, Any]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        chunk_text = content[start:end].strip()
        if not chunk_text:
            continue
        start_line = content[:start].count("\n") + 1
        symbol_match = re.search(
            r"(?:def|class)\s+(\w+)", m.group(),
        ) or re.search(
            r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=)", m.group(),
        )
        symbol = (symbol_match.group(1) or symbol_match.group(2)
                  if symbol_match else None)
        chunks.append({
            "file_path": file_path,
            "content": chunk_text,
            "start_line": start_line,
            "symbol": symbol,
        })

    return chunks


# ---------------------------------------------------------------------------
# Tree-sitter chunking
# ---------------------------------------------------------------------------

def _chunk_by_tree_sitter(
    content: str, file_path: str, language: Any,
) -> list[dict[str, Any]]:
    """Split *content* at function/class boundaries using tree-sitter."""
    from tree_sitter import Parser

    parser = Parser(language)
    content_bytes = content.encode()

    try:
        tree = parser.parse(content_bytes)
    except Exception:
        logger.debug("tree-sitter parse failed for %s, falling back", file_path)
        return _fallback_chunk(content, file_path)

    root = tree.root_node

    # Collect function/class definitions at the top level (and nested)
    def_nodes: list[dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if node.type in ("function_definition", "class_definition",
                         "method_definition", "function_declaration",
                         "class_declaration", "method_declaration",
                         "arrow_function", "generator_function",
                         "function_expression"):
            name_node = node.child_by_field_name("name")
            name = name_node.text.decode() if name_node else None
            start_byte = node.start_byte
            end_byte = node.end_byte
            def_nodes.append({
                "start_byte": start_byte,
                "end_byte": end_byte,
                "start_point": node.start_point,
                "name": name,
                "type": node.type,
            })
        for child in node.children:
            _walk(child)

    _walk(root)

    if not def_nodes:
        return _fallback_chunk(content, file_path)

    # Sort by start_byte
    def_nodes.sort(key=lambda d: d["start_byte"])

    chunks: list[dict[str, Any]] = []
    for i, dn in enumerate(def_nodes):
        start = dn["start_byte"]
        end = def_nodes[i + 1]["start_byte"] if i + 1 < len(def_nodes) else len(content_bytes)
        chunk_text = content_bytes[start:end].decode().strip()
        if not chunk_text:
            continue
        start_line = dn["start_point"][0] + 1
        chunks.append({
            "file_path": file_path,
            "content": chunk_text,
            "start_line": start_line,
            "symbol": dn["name"],
        })

    return chunks


def _fallback_chunk(content: str, file_path: str) -> list[dict[str, Any]]:
    """Whole-file chunk when tree-sitter can't parse."""
    text = content.strip()
    if not text:
        return []
    return [{
        "file_path": file_path,
        "content": text,
        "start_line": 1,
        "symbol": Path(file_path).stem,
    }]


# ---------------------------------------------------------------------------
# BM25Searcher — the in-process tool callable
# ---------------------------------------------------------------------------

class BM25Searcher:
    """In-process code search using BM25Okapi lexical retrieval.

    Implements the contract expected by ``InProcessClient``:
    ``list_tools()`` and ``call_tool(name, **params)``.
    """

    def __init__(self) -> None:
        self._bm25: Any = None          # BM25Okapi instance
        self._chunk_store: list[dict[str, Any]] = []
        self._corpus_tokens: list[list[str]] = []
        self._tokenizer = _whitespace_tokenize

    # -- Tool contract -------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "rag_index",
                "description": (
                    "Walk a directory of source-code files, chunk code by "
                    "function/class boundaries via tree-sitter, tokenize with "
                    "whitespace tokenizer, and build a BM25Okapi index."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Root directory containing source files to index.",
                        },
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "code_search",
                "description": (
                    "Search indexed code using BM25Okapi scoring with a "
                    "whitespace tokenizer."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top results to return.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        ]

    def call_tool(self, name: str, **params: Any) -> Any:
        if name == "rag_index":
            return self._index(params.get("path", ""))
        if name == "code_search":
            return self._search(
                query=params.get("query", ""),
                top_k=int(params.get("top_k", 5)),
            )
        raise ValueError(f"Unknown tool: {name}")

    # -- Index ---------------------------------------------------------------

    def _index(self, path: str) -> dict[str, Any]:
        """Walk *path*, chunk code files, tokenize, and build BM25 index."""
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Not a directory: {root}")

        t0 = time.perf_counter()
        chunks: list[dict[str, Any]] = []
        tokenized_corpus: list[list[str]] = []
        file_count = 0

        for filepath in root.rglob("*"):
            if not filepath.is_file():
                continue
            suffix = filepath.suffix.lower()
            if suffix not in _CODE_EXTENSIONS:
                continue

            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if not content.strip():
                continue

            file_count += 1
            rel_path = str(filepath.relative_to(root))

            # Chunk by function/class
            file_chunks = self._chunk_file(content, rel_path, suffix)
            for ch in file_chunks:
                tokens = self._tokenizer(ch["content"])
                if not tokens:
                    continue
                chunks.append(ch)
                tokenized_corpus.append(tokens)

        if not tokenized_corpus:
            elapsed = time.perf_counter() - t0
            logger.warning("No code chunks found in %s", root)
            return {
                "status": "ok",
                "files_indexed": file_count,
                "chunks_indexed": 0,
                "index_time_s": round(elapsed, 2),
            }

        # Build BM25Okapi index
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._chunk_store = chunks
        self._corpus_tokens = tokenized_corpus

        elapsed = time.perf_counter() - t0
        logger.info(
            "BM25 index built: %d files → %d chunks in %.1fs",
            file_count, len(chunks), elapsed,
        )

        return {
            "status": "ok",
            "files_indexed": file_count,
            "chunks_indexed": len(chunks),
            "index_time_s": round(elapsed, 2),
        }

    # -- Search --------------------------------------------------------------

    def _search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search the BM25 index for chunks most relevant to *query*."""
        if self._bm25 is None or not self._chunk_store:
            return []

        query_tokens = self._tokenizer(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Pair (index, score) and sort descending
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        for idx, score in indexed_scores:
            if len(results) >= top_k:
                break
            chunk = self._chunk_store[idx]
            fp = chunk.get("file_path", "")

            # Deduplicate by file_path: keep highest-scoring chunk per file
            if fp in seen_paths:
                continue
            seen_paths.add(fp)

            results.append({
                "file_path": fp,
                "score": float(score),
                "content": chunk.get("content", "")[:500],
                "symbol": chunk.get("symbol"),
                "start_line": chunk.get("start_line"),
            })

        return results

    # -- Chunking (public for testing) ---------------------------------------

    def _chunk_file(
        self, content: str, file_path: str, suffix: str,
    ) -> list[dict[str, Any]]:
        """Chunk a single file's content.

        Uses tree-sitter when a language is available; falls back to regex
        for Python/JS; whole-file chunk for everything else.
        """
        lang = _get_language(suffix)
        if lang is not None:
            return _chunk_by_tree_sitter(content, file_path, lang)
        # Fallback to regex for Python and JS
        if suffix == ".py":
            return _chunk_by_regex(content, file_path, _PY_DEF_PATTERN)
        if suffix in (".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs"):
            return _chunk_by_regex(content, file_path, _JS_DEF_PATTERN)
        # Generic: whole file as one chunk
        return _fallback_chunk(content, file_path)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _whitespace_tokenize(text: str) -> list[str]:
    """Tokenize text by splitting on whitespace, lowercasing, and filtering.

    Returns a list of tokens (words). Empty tokens and pure-numeric tokens
    are excluded.
    """
    tokens: list[str] = []
    for token in text.split():
        token = token.lower()
        # Strip surrounding punctuation but keep internal dots/underscores
        token = token.strip(".,;:!?()[]{}'\"\\/")
        if not token:
            continue
        # Skip pure-numeric tokens (they add noise to BM25)
        if token.isdigit():
            continue
        tokens.append(token)
    return tokens
