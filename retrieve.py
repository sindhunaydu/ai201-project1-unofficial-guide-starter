"""
retrieve.py — Stage 4: Semantic Retrieval from ChromaDB
========================================================
Architecture position (from planning.md):

  embed_store.py  →  chroma_db/
  retrieve.py     →  RetrievalResult list   ← THIS FILE
  (generation)    →  answer

This module exposes:

  retrieve(query, top_k, ...)  →  list[RetrievalResult]

and a __main__ block for interactive CLI testing.

Retrieval strategy (from planning.md):
  - Semantic search with all-MiniLM-L6-v2 embeddings
  - Top-k: 4–6 chunks (configurable, default 5)
  - Cosine similarity (via normalised dot product in ChromaDB)
  - Optional tag filter to narrow results by topic domain
  - Optional score threshold to drop low-confidence chunks

Run interactive test:
    python retrieve.py
    python retrieve.py --query "How do I apply for OPT?" --top-k 5
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL       = "all-MiniLM-L6-v2"
COLLECTION_NAME   = "intl_student_guide"
CHROMA_DIR        = Path("chroma_db")
DEFAULT_TOP_K     = 5          # planning.md: 4–6 chunks
MIN_TOP_K         = 4
MAX_TOP_K         = 6
DEFAULT_THRESHOLD = 0.30       # cosine similarity floor; drops off-topic results


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    rank:        int
    chunk_id:    str
    score:       float          # cosine similarity in [0, 1]
    text:        str
    source_id:   int
    source_name: str
    source_url:  str
    tags:        list[str]
    token_count: int
    chunk_index: int


# ── Lazy singletons (loaded once per process) ─────────────────────────────────
_model:      Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model '%s' …", EMBED_MODEL)
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"ChromaDB directory not found: {CHROMA_DIR}\n"
                "Run embed_store.py first."
            )
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        collections = [c.name for c in client.list_collections()]
        if COLLECTION_NAME not in collections:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found in {CHROMA_DIR}.\n"
                "Run embed_store.py first."
            )
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=None,   # we supply vectors ourselves
        )
        log.info(
            "Connected to collection '%s' (%d vectors)",
            COLLECTION_NAME, _collection.count(),
        )
    return _collection


# ── Core retrieval function ───────────────────────────────────────────────────

def retrieve(
    query:      str,
    top_k:      int  = DEFAULT_TOP_K,
    threshold:  float = DEFAULT_THRESHOLD,
    tag_filter: Optional[str] = None,
) -> list[RetrievalResult]:
    """
    Retrieve the top-k most relevant chunks for a natural-language query.

    Parameters
    ----------
    query : str
        The user's question or search string.
    top_k : int
        Number of chunks to return. Clamped to [MIN_TOP_K, MAX_TOP_K] = [4, 6]
        as specified in planning.md.
    threshold : float
        Minimum cosine similarity to include a result. Chunks below this score
        are dropped. Default 0.30 — low enough to include paraphrased matches,
        high enough to exclude clearly off-topic chunks.
    tag_filter : str, optional
        If provided, only chunks whose tags contain this string are considered.
        Example: "taxes" → only chunks tagged with "taxes".

    Returns
    -------
    list[RetrievalResult]
        Ranked list (best match first) of matching chunks with full metadata.
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    top_k = max(MIN_TOP_K, min(top_k, MAX_TOP_K))

    model      = _get_model()
    collection = _get_collection()

    # Encode query with the same normalisation used at index time
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    # Build optional ChromaDB where-clause for tag filtering
    # Tags are stored as comma-separated strings, so we use $contains
    where: Optional[dict] = None
    if tag_filter:
        where = {"tags": {"$contains": tag_filter}}

    # Fetch more than top_k so we have headroom after threshold filtering
    fetch_n = min(top_k * 3, collection.count())

    raw = collection.query(
        query_embeddings = query_vec,
        n_results        = fetch_n,
        where            = where,
        include          = ["documents", "metadatas", "distances"],
    )

    results: list[RetrievalResult] = []
    rank = 1

    for doc, meta, dist in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        # ChromaDB returns cosine distance; convert to similarity
        score = round(1.0 - float(dist), 4)

        if score < threshold:
            continue

        results.append(RetrievalResult(
            rank        = rank,
            chunk_id    = meta.get("chunk_id", ""),
            score       = score,
            text        = doc,
            source_id   = int(meta["source_id"]),
            source_name = meta["source_name"],
            source_url  = meta["source_url"],
            tags        = meta["tags"].split(","),     # deserialise tags
            token_count = int(meta["token_count"]),
            chunk_index = int(meta["chunk_index"]),
        ))
        rank += 1

        if len(results) >= top_k:
            break

    return results


# ── Convenience helpers ───────────────────────────────────────────────────────

def retrieve_for_generation(
    query:      str,
    top_k:      int   = DEFAULT_TOP_K,
    threshold:  float = DEFAULT_THRESHOLD,
    tag_filter: Optional[str] = None,
) -> str:
    """
    Returns a single formatted context string ready to be inserted into a
    generation prompt.  Each chunk is separated by a divider with its source.

    Usage in a prompt:
        context = retrieve_for_generation("What is CPT?")
        prompt  = SYSTEM_PROMPT + f"\\n\\nContext:\\n{context}\\n\\nQuestion: {query}"
    """
    chunks = retrieve(query, top_k=top_k, threshold=threshold, tag_filter=tag_filter)
    if not chunks:
        return "No relevant information found."

    parts = []
    for r in chunks:
        parts.append(
            f"[Source: {r.source_name} | {r.source_url}]\n{r.text}"
        )
    return "\n\n---\n\n".join(parts)


def retrieve_as_json(
    query:     str,
    top_k:     int   = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """Returns retrieval results as a JSON string (useful for logging / testing)."""
    results = retrieve(query, top_k=top_k, threshold=threshold)
    return json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

EVAL_QUESTIONS = [
    "Can an international student on F-1 work off-campus during their first year?",
    "Do international students need to file taxes if they have no income?",
    "How can an international student rent an apartment without a credit history?",
    "What should an international student do if they feel unsafe on campus at night?",
    "What are common signs of culture shock and how do students cope with it?",
    "How does an international student open a bank account in the US?",
]


def _print_results(results: list[RetrievalResult], query: str) -> None:
    print(f"\n{'='*68}")
    print(f"Query : {query}")
    print(f"Hits  : {len(results)}")
    print(f"{'='*68}")
    for r in results:
        print(f"\n  Rank {r.rank}  |  score={r.score:.4f}  |  {r.source_name}")
        print(f"  URL   : {r.source_url}")
        print(f"  Tags  : {', '.join(r.tags)}  |  tokens={r.token_count}  |  chunk={r.chunk_index}")
        print(f"  {'─'*60}")
        # Print first 300 chars of chunk text
        preview = r.text[:300].replace("\n", " ")
        print(f"  {preview}{'…' if len(r.text) > 300 else ''}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the International Student Guide vector store"
    )
    parser.add_argument("--query",      type=str,   default=None,
                        help="Question to retrieve chunks for")
    parser.add_argument("--top-k",      type=int,   default=DEFAULT_TOP_K,
                        help=f"Chunks to return (default {DEFAULT_TOP_K}, range {MIN_TOP_K}–{MAX_TOP_K})")
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                        help=f"Min similarity score (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--tag",        type=str,   default=None,
                        help="Filter results to a specific tag (e.g. taxes, housing)")
    parser.add_argument("--eval",       action="store_true",
                        help="Run all 6 eval questions from planning.md")
    args = parser.parse_args()

    if args.eval:
        print("\nRunning all 6 evaluation questions from planning.md …")
        for q in EVAL_QUESTIONS:
            results = retrieve(q, top_k=args.top_k, threshold=args.threshold)
            _print_results(results, q)
    elif args.query:
        results = retrieve(
            args.query,
            top_k=args.top_k,
            threshold=args.threshold,
            tag_filter=args.tag,
        )
        _print_results(results, args.query)
    else:
        # Default: run two sample queries to show it works
        samples = [
            "What is OPT and when should I apply?",
            "What documents do I need to open a bank account?",
        ]
        for q in samples:
            results = retrieve(q, top_k=DEFAULT_TOP_K, threshold=DEFAULT_THRESHOLD)
            _print_results(results, q)


if __name__ == "__main__":
    main()