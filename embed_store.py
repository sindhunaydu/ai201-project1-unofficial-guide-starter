"""
embed_store.py — Stage 3: Embed Chunks and Store in ChromaDB
=============================================================
Architecture position (from planning.md):

  fetch_raw.py        →  raw/text/*.txt
  ingest_and_chunk.py →  output/chunks.json
  embed_store.py      →  chroma_db/          ← THIS FILE
  retrieve.py         →  query results

What this script does:
  1. Loads output/chunks.json produced by ingest_and_chunk.py
  2. Embeds every chunk with sentence-transformers/all-MiniLM-L6-v2
  3. Stores vectors + full source metadata in a persistent ChromaDB collection
  4. Prints a storage summary and runs a smoke-test query

Run:
    python embed_store.py

Outputs:
    chroma_db/          persistent ChromaDB directory
    output/embed_report.json   per-chunk embedding report
"""

import json
import logging
import time
from pathlib import Path

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
EMBED_MODEL      = "all-MiniLM-L6-v2"
COLLECTION_NAME  = "intl_student_guide"
CHUNKS_PATH      = Path("output/chunks.json")
CHROMA_DIR       = Path("chroma_db")
REPORT_PATH      = Path("output/embed_report.json")
BATCH_SIZE       = 64          # chunks per embedding batch
SMOKE_TEST_QUERY = "How do I apply for OPT after graduation?"

# ── Metadata keys stored alongside every vector ───────────────────────────────
# ChromaDB metadata must be flat (str | int | float | bool).
# Lists (tags) are serialised to comma-separated strings and deserialised
# on the way out in retrieve.py.
META_KEYS = ["source_id", "source_name", "source_url", "tags",
             "token_count", "chunk_index"]


# ── Step 1 — Load chunks ──────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {path}\n"
            "Run ingest_and_chunk.py first."
        )
    chunks = json.loads(path.read_text(encoding="utf-8"))
    log.info("Loaded %d chunks from %s", len(chunks), path)
    return chunks


# ── Step 2 — Build ChromaDB collection ───────────────────────────────────────

def get_or_create_collection(chroma_dir: Path) -> tuple[chromadb.Collection, chromadb.Client]:
    """
    Create a persistent ChromaDB client and return the collection.
    If the collection already exists it is cleared and rebuilt from scratch
    so this script is safely re-runnable (idempotent).

    We use chromadb's built-in embedding function wrapper set to None so that
    we supply our own pre-computed vectors — this gives us full control over
    the embedding model and batching strategy.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    # Drop and recreate for a clean rebuild
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        log.info("Dropped existing collection '%s' for clean rebuild", COLLECTION_NAME)

    # embedding_function=None means we supply vectors directly via add()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "International Student Survival Guide — 15 sources",
            "embed_model":  EMBED_MODEL,
            "hnsw:space":   "cosine",   # cosine similarity for sentence embeddings
        },
        embedding_function=None,
    )
    log.info("Created collection '%s' (cosine space)", COLLECTION_NAME)
    return collection, client


# ── Step 3 — Embed and upsert in batches ─────────────────────────────────────

def embed_and_store(
    chunks: list[dict],
    collection: chromadb.Collection,
    model: SentenceTransformer,
) -> list[dict]:
    """
    Embed all chunks in batches and upsert to ChromaDB.
    Returns a report list (one dict per chunk).
    """
    total   = len(chunks)
    report  = []
    t_start = time.perf_counter()

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        batch_end = batch_start + len(batch)
        log.info("  Embedding batch %d–%d / %d …", batch_start + 1, batch_end, total)

        texts = [c["text"] for c in batch]

        # sentence-transformers encode returns a numpy array (batch, dim)
        # convert_to_numpy=True keeps it CPU-friendly; normalise for cosine
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,   # unit vectors → cosine = dot product
            show_progress_bar=False,
        )

        ids        = [c["chunk_id"]   for c in batch]
        documents  = texts            # ChromaDB stores original text as "document"
        metadatas  = [
            {
                "source_id":   c["source_id"],
                "source_name": c["source_name"],
                "source_url":  c["source_url"],
                "tags":        ",".join(c["tags"]),   # flatten list → str
                "token_count": c["token_count"],
                "chunk_index": c["chunk_index"],
            }
            for c in batch
        ]

        collection.add(
            ids        = ids,
            embeddings = embeddings.tolist(),   # ChromaDB expects plain lists
            documents  = documents,
            metadatas  = metadatas,
        )

        for i, chunk in enumerate(batch):
            report.append({
                "chunk_id":    chunk["chunk_id"],
                "source_name": chunk["source_name"],
                "token_count": chunk["token_count"],
                "embed_dim":   int(embeddings.shape[1]),
                "norm":        round(float((embeddings[i] ** 2).sum() ** 0.5), 6),
            })

    elapsed = time.perf_counter() - t_start
    log.info(
        "Embedded and stored %d chunks in %.1fs  (%.0f chunks/s)",
        total, elapsed, total / elapsed if elapsed > 0 else 0,
    )
    return report


# ── Step 4 — Smoke test ───────────────────────────────────────────────────────

def smoke_test(collection: chromadb.Collection, model: SentenceTransformer) -> None:
    """Query the freshly built collection with one question to verify end-to-end."""
    log.info("\nSmoke test query: '%s'", SMOKE_TEST_QUERY)

    q_vec = model.encode(
        [SMOKE_TEST_QUERY],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    results = collection.query(
        query_embeddings=q_vec,
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    log.info("Top-3 results:")
    for rank, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ), 1):
        score = round(1 - dist, 4)   # cosine similarity = 1 - cosine distance
        preview = doc[:120].replace("\n", " ")
        log.info(
            "  [%d] score=%.4f  source=%s\n"
            "       %s…",
            rank, score, meta["source_name"], preview,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=" * 60)
    log.info("embed_store.py  —  Stage 3: Embed + Store")
    log.info("Model      : %s", EMBED_MODEL)
    log.info("Collection : %s", COLLECTION_NAME)
    log.info("ChromaDB   : %s", CHROMA_DIR)
    log.info("=" * 60)

    # 1. Load
    chunks = load_chunks(CHUNKS_PATH)

    # 2. Load embedding model
    log.info("Loading embedding model '%s' …", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    log.info(
        "Model ready — embedding dim: %d",
        model.get_sentence_embedding_dimension(),
    )

    # 3. ChromaDB
    collection, _ = get_or_create_collection(CHROMA_DIR)

    # 4. Embed + store
    report = embed_and_store(chunks, collection, model)

    # 5. Verify count
    stored = collection.count()
    log.info("Verified: %d vectors in collection", stored)
    assert stored == len(chunks), (
        f"Count mismatch: stored {stored}, expected {len(chunks)}"
    )

    # 6. Smoke test
    smoke_test(collection, model)

    # 7. Save report
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("\nEmbed report saved → %s", REPORT_PATH)
    log.info("=" * 60)
    log.info("Stage 3 complete.  Next: use retrieve.py to query the store")


if __name__ == "__main__":
    main()