"""
International Student Survival Guide — Document Ingestion & Chunking Pipeline
==============================================================================
Spec (from planning.md):
  - Chunk size  : 500 tokens
  - Overlap     : 100 tokens
  - Strategy    : Semantic / paragraph-aware chunking
  - Embed model : all-MiniLM-L6-v2  (sentence-transformers)
  - Sources     : 15 URLs listed in the planning document

Stage ordering
--------------
1. fetch_raw.py        — download HTML, save raw text to raw/text/*.txt
2. ingest_and_chunk.py — read raw/text/ → clean → chunk → output/chunks.json

This script reads raw text files written by fetch_raw.py.
If raw/manifest.json is missing it falls back to live fetching (dev mode).
"""

import re
import json
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # tokens
CHUNK_OVERLAP = 100   # tokens
EMBED_MODEL   = "all-MiniLM-L6-v2"   # sentence-transformers model name

RAW_TEXT_DIR  = Path("raw/text")
MANIFEST_PATH = Path("raw/manifest.json")
OUTPUT_DIR    = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load model once — tokenizer is used for token counting
_model    = SentenceTransformer(EMBED_MODEL)
tokenizer = _model.tokenizer

# ── Source catalogue ──────────────────────────────────────────────────────────
SOURCES = [
    {"id":  1, "name": "DHS – Maintaining F-1 Status",
     "url": "https://studyinthestates.dhs.gov/students",
     "tags": ["immigration", "f1", "status"]},

    {"id":  2, "name": "USCIS – Student Employment (OPT/CPT)",
     "url": "https://www.uscis.gov/working-in-the-united-states/students-and-exchange-visitors",
     "tags": ["employment", "opt", "cpt"]},

    {"id":  3, "name": "SSA – Social Security Number",
     "url": "https://bechtel.stanford.edu/navigate-international-life/social-security-number-ssn",
     "tags": ["ssn", "social-security"]},

    {"id":  4, "name": "IRS – Tax Info for International Students",
     "url": "https://www.irs.gov/individuals/international-taxpayers",
     "tags": ["taxes", "irs"]},

    {"id":  5, "name": "Healthcare.gov – Student Health Insurance",
     "url": "https://www.healthcare.gov/young-adults/college-students/",
     "tags": ["health", "insurance"]},

    {"id":  6, "name": "CFPB – U.S. Banking Basics",
     "url": "https://www.consumerfinance.gov/consumer-tools/bank-accounts/",
     "tags": ["banking", "finance"]},

    {"id":  7, "name": "myFICO – Credit Score Education",
     "url": "https://www.myfico.com/credit-education",
     "tags": ["credit", "finance"]},

    {"id":  8, "name": "FTC – Renting Apartments Guide",
     "url": "https://consumer.ftc.gov/articles/tenant-background-checks-and-your-rights",
     "tags": ["housing", "renting"]},

    {"id":  9, "name": "Ready.gov – Campus Safety",
     "url": "https://www.ready.gov/campus",
     "tags": ["safety", "campus"]},

    {"id": 10, "name": "Reddit – r/IntltoUSA Discussions",
     "url": "https://www.reddit.com/r/IntltoUSA/",
     "tags": ["community", "reddit"]},

    {"id": 11, "name": "UC Berkeley – Career Services (International)",
     "url": "https://career.berkeley.edu/communities/international-students/",
     "tags": ["career", "internship"]},

    {"id": 12, "name": "UIUC – Academic Integrity / Plagiarism",
     "url": "https://provost.illinois.edu/policies/policies/academic-integrity/students-quick-reference-guide-to-academic-integrity/",
     "tags": ["academics", "integrity"]},

    {"id": 13, "name": "UC Berkeley – Internship & Career Preparation",
     "url": "https://career.berkeley.edu/communities/international-students/",
     "tags": ["career", "internship"]},

    {"id": 14, "name": "Stanford – Taxes",
     "url": "https://bechtel.stanford.edu/navigate-international-life/taxes",
     "tags": ["taxes"]},

    {"id": 15, "name": "USC – Culture Shock Guide",
     "url": "https://ois.usc.edu/2024/04/08/navigating-culture-shock-a-guide-for-international-students-in-the-u-s/",
     "tags": ["culture", "adjustment"]},
]

# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    chunk_id:    str
    source_id:   int
    source_name: str
    source_url:  str
    tags:        list[str]
    text:        str
    token_count: int
    chunk_index: int   # position within the source document

# ── Step 1 — Fetch & clean ────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; IntlStudentGuideBot/1.0; "
        "+https://example.com/bot)"
    )
}

# Tags whose text we always discard (navigation, cookie banners, etc.)
_NOISE_TAGS = {
    "nav", "header", "footer", "aside", "script", "style",
    "noscript", "form", "button", "iframe", "figure", "figcaption",
}

# Patterns for leftover junk lines
_JUNK_PATTERNS = [
    re.compile(r"^\s*$"),                          # blank lines
    re.compile(r"skip (to|main)", re.I),           # skip-nav links
    re.compile(r"cookie(s| policy| notice)", re.I),
    re.compile(r"javascript must be enabled", re.I),
    re.compile(r"^\W{1,3}$"),                      # lone punctuation
    re.compile(r"^(menu|search|home|login|sign in|subscribe)$", re.I),
]


def _is_junk_line(line: str) -> bool:
    return any(p.search(line) for p in _JUNK_PATTERNS)


def _clean_text(raw_text: str) -> Optional[str]:
    """
    Clean raw extracted text saved by fetch_raw.py.
    Strips junk lines (nav, cookie banners, blank lines, lone punctuation),
    collapses whitespace, returns paragraph-structured plain text.
    """
    lines = raw_text.splitlines()
    blocks: list[str] = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line and not _is_junk_line(line):
            blocks.append(line)

    text = "\n\n".join(blocks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = text.strip()
    return text if text else None


# ── Step 2 — Semantic / paragraph-aware chunking ──────────────────────────────

def _token_count(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines, keeping each non-empty paragraph as a unit."""
    return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]


def _split_paragraph_into_sentences(para: str) -> list[str]:
    """Naive sentence splitter — good enough for the document types in use."""
    # Split on . ! ? followed by whitespace + capital letter (or end)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'\(])", para)
    return [s.strip() for s in parts if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int  = CHUNK_SIZE,
    overlap:    int  = CHUNK_OVERLAP,
) -> list[str]:
    """
    Semantic chunking strategy
    --------------------------
    1. Split the document into paragraphs (blank-line boundaries).
    2. Add paragraphs to the current chunk until we reach `chunk_size` tokens.
    3. When a paragraph would overflow, try adding individual sentences instead.
    4. When the current chunk is full (or a sentence still overflows), close it
       and start a new chunk seeded with the last `overlap` tokens of content.

    This keeps semantically related paragraphs together while respecting the
    500-token limit and 100-token overlap specified in planning.md.
    """
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[str] = []

    # running buffer of text units that will form the current chunk
    current_units: list[str] = []
    current_tokens: int = 0

    def flush_chunk() -> list[str]:
        """Close the current chunk and return seed units for the next one."""
        nonlocal current_units, current_tokens

        chunk_text_ = " ".join(current_units).strip()
        if chunk_text_:
            chunks.append(chunk_text_)

        # Build overlap seed: take units from the end until we hit overlap tokens
        seed_units: list[str] = []
        seed_tokens = 0
        for unit in reversed(current_units):
            ut = _token_count(unit)
            if seed_tokens + ut > overlap:
                break
            seed_units.insert(0, unit)
            seed_tokens += ut

        current_units  = seed_units
        current_tokens = seed_tokens
        return seed_units

    def add_unit(unit: str) -> None:
        nonlocal current_tokens
        current_units.append(unit)
        current_tokens += _token_count(unit)

    for para in paragraphs:
        para_tokens = _token_count(para)

        # Case A: paragraph fits entirely
        if current_tokens + para_tokens <= chunk_size:
            add_unit(para)

        # Case B: paragraph doesn't fit — try sentence-by-sentence
        else:
            sentences = _split_paragraph_into_sentences(para)

            for sent in sentences:
                sent_tokens = _token_count(sent)

                if current_tokens + sent_tokens <= chunk_size:
                    add_unit(sent)
                else:
                    # Flush what we have, then handle the sentence
                    if current_units:
                        flush_chunk()

                    if sent_tokens <= chunk_size:
                        add_unit(sent)
                    else:
                        # Sentence itself is too long — hard-split on token windows
                        token_ids = tokenizer.encode(
                            sent, add_special_tokens=False
                        )
                        start = 0
                        while start < len(token_ids):
                            end   = min(start + chunk_size, len(token_ids))
                            piece = tokenizer.decode(
                                token_ids[start:end],
                                skip_special_tokens=True,
                            )
                            chunks.append(piece.strip())
                            start += chunk_size - overlap

    # Final flush
    if current_units:
        remaining = " ".join(current_units).strip()
        if remaining:
            chunks.append(remaining)

    return [c for c in chunks if c]


# ── Step 3 — Load raw text ───────────────────────────────────────────────────

def _load_raw_text_from_manifest() -> dict[int, str]:
    """
    Read raw/manifest.json (written by fetch_raw.py) and return a mapping of
    source_id -> raw text content.  Only "ok" entries are included.
    """
    if not MANIFEST_PATH.exists():
        return {}

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw_texts: dict[int, str] = {}

    for entry in manifest:
        if entry.get("status") != "ok":
            log.warning("  Skipping src %02d in manifest (status=%s)",
                        entry["id"], entry.get("status"))
            continue
        text_path = Path(entry["text_path"])
        if not text_path.exists():
            log.warning("  Raw text file missing: %s", text_path)
            continue
        raw_texts[entry["id"]] = text_path.read_text(encoding="utf-8")
        log.info("  Loaded raw text for src %02d  (%d chars)",
                 entry["id"], len(raw_texts[entry["id"]]))

    return raw_texts


# ── Step 4 — Orchestrate everything ──────────────────────────────────────────

def run_pipeline() -> list[Chunk]:
    all_chunks: list[Chunk] = []

    # Prefer pre-fetched raw text files; fall back to live fetch if absent
    if MANIFEST_PATH.exists():
        log.info("Manifest found → reading from %s", RAW_TEXT_DIR)
        raw_text_map = _load_raw_text_from_manifest()
        use_manifest = True
    else:
        log.warning("No manifest found at %s — falling back to live fetch", MANIFEST_PATH)
        log.warning("Run fetch_raw.py first for reproducible results.")
        raw_text_map = {}
        use_manifest = False

    for src in SOURCES:
        log.info("→ [%2d/15] %s", src["id"], src["name"])

        # ── Acquire raw text ─────────────────────────────────────────────
        if use_manifest:
            raw_text = raw_text_map.get(src["id"])
            if not raw_text:
                log.warning("  ! Not in manifest or file missing — skipping.")
                continue
            source_label = f"raw/text (src {src['id']:02d})"
        else:
            log.info("         Fetching live: %s", src["url"])
            raw_text = raw_text_map.get(src["id"])   # will be None
            if raw_text is None:
                # Live fallback: fetch + minimal extraction (no cleaning yet)
                try:
                    resp = requests.get(src["url"], headers=HEADERS, timeout=25)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.content, "html.parser")
                    for tag in soup.find_all(["script", "style", "noscript"]):
                        tag.decompose()
                    raw_text = soup.get_text(separator="\n")
                    time.sleep(1.5)
                except Exception as exc:
                    log.warning("  ! Fetch failed: %s — skipping.", exc)
                    continue
            source_label = f"live fetch"

        log.info("  Raw text: %d chars  [%s]", len(raw_text), source_label)

        # ── Clean ────────────────────────────────────────────────────────
        cleaned = _clean_text(raw_text)
        if not cleaned:
            log.warning("  ! Nothing left after cleaning — skipping.")
            continue
        log.info("  Cleaned:  %d chars", len(cleaned))

        # ── Chunk ────────────────────────────────────────────────────────
        text_chunks = chunk_text(cleaned)
        log.info("  Produced  %d chunks", len(text_chunks))

        for idx, text in enumerate(text_chunks):
            tc = _token_count(text)
            chunk = Chunk(
                chunk_id    = f"src{src['id']:02d}_chunk{idx:04d}",
                source_id   = src["id"],
                source_name = src["name"],
                source_url  = src["url"],
                tags        = src["tags"],
                text        = text,
                token_count = tc,
                chunk_index = idx,
            )
            all_chunks.append(chunk)

    return all_chunks


# ── Step 4 — Persist results ──────────────────────────────────────────────────

def save_results(chunks: list[Chunk]) -> None:
    # 1. Full JSON dump
    json_path = OUTPUT_DIR / "chunks.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, indent=2, ensure_ascii=False)
    log.info("Saved %d chunks → %s", len(chunks), json_path)

    # 2. Human-readable plain-text preview
    preview_path = OUTPUT_DIR / "chunks_preview.txt"
    with preview_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(f"{'='*70}\n")
            f.write(f"ID     : {c.chunk_id}\n")
            f.write(f"Source : {c.source_name}\n")
            f.write(f"URL    : {c.source_url}\n")
            f.write(f"Tags   : {', '.join(c.tags)}\n")
            f.write(f"Tokens : {c.token_count}\n")
            f.write(f"Index  : {c.chunk_index}\n")
            f.write(f"{'-'*70}\n")
            f.write(c.text + "\n\n")
    log.info("Saved preview     → %s", preview_path)

    # 3. Stats summary
    if chunks:
        token_counts = [c.token_count for c in chunks]
        stats = {
            "total_chunks"  : len(chunks),
            "sources_hit"   : len({c.source_id for c in chunks}),
            "min_tokens"    : min(token_counts),
            "max_tokens"    : max(token_counts),
            "avg_tokens"    : round(sum(token_counts) / len(token_counts), 1),
        }
        stats_path = OUTPUT_DIR / "stats.json"
        with stats_path.open("w") as f:
            json.dump(stats, f, indent=2)
        log.info("Stats: %s", stats)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting ingestion pipeline")
    log.info("Config: chunk_size=%d  overlap=%d  model=%s",
             CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL)

    chunks = run_pipeline()
    save_results(chunks)

    log.info("Done. Total chunks produced: %d", len(chunks))