"""Naive embedding-only RAG baseline.

Simple pipeline: chunk code by function/class boundaries → embed with
sentence-transformers (all-MiniLM-L6-v2) → cosine similarity top-K via FAISS.

No FTS5, no code graph, no symbol re-ranking. Serves as the baseline showing
the value of a full code-intelligence layer.
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# File extensions that are considered "code" for the baseline.
_CODE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt",
    ".scala", ".cs", ".m", ".mm",
}


def _is_code_file(filepath: str | Path) -> bool:
    """Return True if the file extension suggests it's a source-code file."""
    return Path(filepath).suffix.lower() in _CODE_EXTENSIONS


# ---------------------------------------------------------------------------
# Code chunking
# ---------------------------------------------------------------------------

# Patterns that mark the start of a logical code block (function, method,
# class, or module-level construct).
_PY_DEF_PATTERN = re.compile(
    r"^(\s*)(?:(?:async\s+)?def\s+\w+|class\s+\w+)",
    re.MULTILINE,
)
_JS_DEF_PATTERN = re.compile(
    r"^(\s*)(?:(?:async\s+)?function\s+\w+|class\s+\w+|"
    r"(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)",
    re.MULTILINE,
)


def _chunk_by_defs(content: str, file_path: str, lang_pattern: re.Pattern) -> list[dict[str, Any]]:
    """Split *content* at function/class definition boundaries.

    Returns a list of dicts with ``file_path``, ``content`` (the chunk text),
    ``start_line`` (1-based), and ``symbol`` (detected def/class name, if any).
    """
    lines = content.split("\n")
    matches = list(lang_pattern.finditer(content))

    if not matches:
        # No definitions found — treat the whole file as one chunk.
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

        # Determine the 1-based start line
        start_line = content[:start].count("\n") + 1

        # Extract symbol name from the match
        symbol_match = re.search(
            r"(?:def|class)\s+(\w+)", m.group(),
        ) or re.search(
            r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=)",
            m.group(),
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


def _split_large_chunk(chunk: dict[str, Any], max_chars: int) -> list[dict[str, Any]]:
    """If *chunk* exceeds *max_chars*, split it at blank-line boundaries."""
    text = chunk["content"]
    if len(text) <= max_chars:
        return [chunk]

    # Split at blank lines
    paragraphs = re.split(r"\n\s*\n", text)
    result: list[dict[str, Any]] = []
    accum = ""
    accum_start = chunk["start_line"]

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if accum and len(accum) + len(para) + 2 > max_chars:
            result.append({
                "file_path": chunk["file_path"],
                "content": accum.strip(),
                "start_line": accum_start,
                "symbol": chunk.get("symbol"),
            })
            accum = para
            # Approximate start_line — good enough for debugging
            accum_start = accum_start + accum.count("\n") + 2
        else:
            if accum:
                accum += "\n\n" + para
            else:
                accum = para

    if accum.strip():
        result.append({
            "file_path": chunk["file_path"],
            "content": accum.strip(),
            "start_line": accum_start,
            "symbol": chunk.get("symbol"),
        })

    return result or [chunk]


# ---------------------------------------------------------------------------
# NaiveRAGSearcher — the in-process tool callable
# ---------------------------------------------------------------------------

class NaiveRAGSearcher:
    """In-process code search using sentence-transformers + FAISS.

    Implements the contract expected by ``InProcessClient``:
    ``list_tools()`` and ``call_tool(name, **params)``.

    Environment variables
    ---------------------
    NAIVE_RAG_MODEL_NAME : str
        HuggingFace sentence-transformers model name (default: ``all-MiniLM-L6-v2``).
    NAIVE_RAG_CHUNK_SIZE : int
        Maximum characters per chunk before splitting (default: 1500).
    NAIVE_RAG_BATCH_SIZE : int
        Batch size for embedding (default: 64).
    """

    def __init__(self) -> None:
        self.model_name = os.getenv("NAIVE_RAG_MODEL_NAME", "all-MiniLM-L6-v2")
        self.max_chunk_chars = int(os.getenv("NAIVE_RAG_CHUNK_SIZE", "1500"))
        self.batch_size = int(os.getenv("NAIVE_RAG_BATCH_SIZE", "64"))

        # Lazy-loaded model and index
        self._model: Any = None
        self._faiss_index: Any = None   # FAISS IndexFlatIP
        self._chunk_store: list[dict[str, Any]] = []  # metadata per chunk
        self._dim: int | None = None

    # -- Tool contract -------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "rag_index",
                "description": (
                    "Index a directory of source-code files. Walks the tree, "
                    "chunks code by function/class boundaries, embeds each chunk "
                    "with sentence-transformers, and builds a FAISS index."
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
                    "Search indexed code using cosine similarity between the "
                    "query embedding and chunk embeddings."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural-language query.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top results to return.",
                            "default": 5,
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional project directory to scope search.",
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
                path=params.get("path"),
            )
        raise ValueError(f"Unknown tool: {name}")

    # -- Model lazy-loading --------------------------------------------------

    def _get_model(self) -> Any:
        if self._model is None:
            import sentence_transformers
            logger.info("Loading sentence-transformers model: %s", self.model_name)
            t0 = time.perf_counter()
            self._model = sentence_transformers.SentenceTransformer(self.model_name)
            logger.info("Model loaded in %.1fs", time.perf_counter() - t0)
        return self._model

    # -- Index ---------------------------------------------------------------

    def _index(self, path: str) -> dict[str, Any]:
        """Walk *path*, chunk code files, embed, and build a FAISS index."""
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Not a directory: {root}")

        t0 = time.perf_counter()

        # 1. Walk files and chunk
        chunks: list[dict[str, Any]] = []
        files_seen: set[str] = set()
        file_count = 0

        for filepath in root.rglob("*"):
            if not filepath.is_file():
                continue
            if not _is_code_file(filepath):
                continue

            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            if not content.strip():
                continue

            file_count += 1
            rel_path = str(filepath.relative_to(root))
            files_seen.add(rel_path)

            file_chunks = self._chunk_file_content(content, rel_path)
            for ch in file_chunks:
                for split in _split_large_chunk(ch, self.max_chunk_chars):
                    chunks.append(split)

        if not chunks:
            chunk_time = time.perf_counter() - t0
            logger.warning("No code chunks found in %s", root)
            return {
                "status": "ok",
                "files_indexed": file_count,
                "chunks_embedded": 0,
                "index_time_s": round(chunk_time, 2),
            }

        logger.info(
            "Chunked %d files → %d chunks in %.1fs",
            file_count, len(chunks), time.perf_counter() - t0,
        )

        # 2. Embed all chunks in batches
        model = self._get_model()
        chunk_texts = [c["content"] for c in chunks]

        embed_start = time.perf_counter()
        embeddings = model.encode(
            chunk_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine via inner product
        )
        embed_time = time.perf_counter() - embed_start

        # 3. Build (or rebuild) FAISS index
        dimension = embeddings.shape[1]
        self._dim = dimension

        import faiss
        self._faiss_index = faiss.IndexFlatIP(dimension)   # inner product = cosine on norm'd vecs
        self._faiss_index.add(embeddings.astype(np.float32))
        self._chunk_store = chunks

        total_time = time.perf_counter() - t0
        logger.info(
            "Index built: %d vectors, dim=%d, in %.1fs (embed: %.1fs)",
            len(chunks), dimension, total_time, embed_time,
        )

        return {
            "status": "ok",
            "files_indexed": file_count,
            "chunks_embedded": len(chunks),
            "index_time_s": round(total_time, 2),
            "embed_time_s": round(embed_time, 2),
        }

    # -- Search --------------------------------------------------------------

    def _search(
        self, query: str, top_k: int = 5, path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the FAISS index for chunks most similar to *query*."""
        if self._faiss_index is None or not self._chunk_store:
            return []

        model = self._get_model()
        query_vec = model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        scores, indices = self._faiss_index.search(query_vec, min(top_k, len(self._chunk_store)))

        results: list[dict[str, Any]] = []
        seen_paths: set[str] = set()

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self._chunk_store):
                continue
            chunk = self._chunk_store[int(idx)]
            fp = chunk.get("file_path", "")

            # Path filtering: naive RAG indexes a single directory, so all
            # results are already scoped. The *path* parameter is accepted for
            # API compatibility with nova-rag's code_search but is a no-op here.
            _ = path  # accepted but unused — single-index architecture

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

    def _chunk_file_content(self, content: str, file_path: str) -> list[dict[str, Any]]:
        """Chunk a single file's content by function/class boundaries.

        Public so tests can exercise chunking directly.
        """
        suffix = Path(file_path).suffix.lower()

        if suffix in (".py",):
            return _chunk_by_defs(content, file_path, _PY_DEF_PATTERN)
        elif suffix in (".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs"):
            return _chunk_by_defs(content, file_path, _JS_DEF_PATTERN)
        else:
            # Generic: whole file as one chunk
            text = content.strip()
            if not text:
                return []
            return [{
                "file_path": file_path,
                "content": text,
                "start_line": 1,
                "symbol": None,
            }]
