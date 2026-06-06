"""
fetch_raw.py — Stage 1: Load & Persist Raw Documents
=====================================================
Fetches all 15 source URLs from the International Student Survival Guide
project and saves two artefacts per document:

  raw/html/<id>_<slug>.html   — exact server response (bytes)
  raw/text/<id>_<slug>.txt    — visible text extracted from HTML, no cleaning

Nothing is cleaned, normalised, or chunked here.
That happens in ingest_and_chunk.py which reads from raw/text/.

Run:
    python fetch_raw.py

Output manifest:  raw/manifest.json
"""

import json
import re
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RAW_HTML_DIR = Path("raw/html")
RAW_TEXT_DIR = Path("raw/text")
MANIFEST_PATH = Path("raw/manifest.json")
CRAWL_DELAY   = 2.0          # seconds between requests (polite crawling)
REQUEST_TIMEOUT = 25         # seconds

RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
RAW_TEXT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; IntlStudentGuideBot/1.0; "
        "research project; contact: your@email.com)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── Source catalogue (15 URLs from planning.md) ───────────────────────────────
SOURCES = [
    {"id":  1, "name": "DHS – Maintaining F-1 Status",
     "url": "https://studyinthestates.dhs.gov/students",
     "tags": ["immigration", "f1", "status"]},

    {"id":  2, "name": "USCIS – Student Employment OPT/CPT",
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

    {"id":  6, "name": "CFPB – US Banking Basics",
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

    {"id": 10, "name": "Reddit – r/IntltoUSA",
     "url": "https://www.reddit.com/r/IntltoUSA/",
     "tags": ["community", "reddit"]},

    {"id": 11, "name": "UC Berkeley – Career Services International",
     "url": "https://career.berkeley.edu/communities/international-students/",
     "tags": ["career", "internship"]},

    {"id": 12, "name": "UIUC – Academic Integrity",
     "url": "https://provost.illinois.edu/policies/policies/academic-integrity/students-quick-reference-guide-to-academic-integrity/",
     "tags": ["academics", "integrity"]},

    {"id": 13, "name": "UC Berkeley – Internship and Career Preparation",
     "url": "https://career.berkeley.edu/communities/international-students/",
     "tags": ["career", "internship"]},

    {"id": 14, "name": "Stanford – Taxes",
     "url": "https://bechtel.stanford.edu/navigate-international-life/taxes",
     "tags": ["taxes"]},

    {"id": 15, "name": "USC – Culture Shock Guide",
     "url": "https://ois.usc.edu/2024/04/08/navigating-culture-shock-a-guide-for-international-students-in-the-u-s/",
     "tags": ["culture", "adjustment"]},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_slug(source_id: int, name: str) -> str:
    """
    Turn  (2, "USCIS – Student Employment OPT/CPT")
    into  "02_uscis_student_employment_opt_cpt"
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)[:60]
    return f"{source_id:02d}_{slug}"


def extract_raw_text(html: str) -> str:
    """
    Extract ALL visible text from HTML with zero cleaning.

    Strategy: use BeautifulSoup's .get_text() with a newline separator so that
    block-level structure (paragraphs, headings, list items) is preserved as
    separate lines. This is intentionally unfiltered — nav bars, footers,
    cookie notices and all. Cleaning happens later in ingest_and_chunk.py.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove only tags that never contain visible text
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    raw_text = soup.get_text(separator="\n")
    return raw_text


def fetch_source(src: dict) -> dict:
    """
    Fetch one URL. Returns a result dict recording what happened.
    Saves raw HTML and raw text to disk.
    """
    slug = make_slug(src["id"], src["name"])
    html_path = RAW_HTML_DIR / f"{slug}.html"
    text_path = RAW_TEXT_DIR / f"{slug}.txt"

    result = {
        "id":        src["id"],
        "name":      src["name"],
        "url":       src["url"],
        "tags":      src["tags"],
        "slug":      slug,
        "html_path": str(html_path),
        "text_path": str(text_path),
        "status":    None,    # "ok" | "http_error" | "network_error"
        "http_code": None,
        "html_bytes": None,
        "raw_text_chars": None,
        "raw_text_lines": None,
        "fetched_at": None,
        "error": None,
    }

    try:
        log.info("  GET %s", src["url"])
        resp = requests.get(src["url"], headers=HEADERS, timeout=REQUEST_TIMEOUT)
        result["http_code"]  = resp.status_code
        result["fetched_at"] = datetime.now(timezone.utc).isoformat()
        resp.raise_for_status()

        # ── Save raw HTML (bytes, exactly as received) ────────────────────
        html_path.write_bytes(resp.content)
        result["html_bytes"] = len(resp.content)
        log.info("  ✓ HTML saved  (%d bytes) → %s", len(resp.content), html_path)

        # ── Extract & save raw text (no cleaning) ─────────────────────────
        html_str  = resp.content.decode(resp.apparent_encoding or "utf-8",
                                        errors="replace")
        raw_text  = extract_raw_text(html_str)
        text_path.write_text(raw_text, encoding="utf-8")
        result["raw_text_chars"] = len(raw_text)
        result["raw_text_lines"] = raw_text.count("\n")
        result["status"] = "ok"
        log.info("  ✓ Text saved  (%d chars, %d lines) → %s",
                 len(raw_text), raw_text.count("\n"), text_path)

    except requests.HTTPError as exc:
        result["status"] = "http_error"
        result["error"]  = str(exc)
        log.warning("  ✗ HTTP error: %s", exc)

    except requests.RequestException as exc:
        result["status"] = "network_error"
        result["error"]  = str(exc)
        log.warning("  ✗ Network error: %s", exc)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("fetch_raw.py  —  Stage 1: Load & Persist Raw Documents")
    log.info("Sources  : %d URLs", len(SOURCES))
    log.info("HTML dir : %s", RAW_HTML_DIR)
    log.info("Text dir : %s", RAW_TEXT_DIR)
    log.info("=" * 60)

    manifest = []

    for i, src in enumerate(SOURCES, 1):
        log.info("\n[%2d / %2d]  %s", i, len(SOURCES), src["name"])
        result = fetch_source(src)
        manifest.append(result)

        # Polite delay between requests (skip after last)
        if i < len(SOURCES):
            log.info("  … waiting %.1fs", CRAWL_DELAY)
            time.sleep(CRAWL_DELAY)

    # ── Write manifest ────────────────────────────────────────────────────────
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("\n" + "=" * 60)
    log.info("Manifest written → %s", MANIFEST_PATH)

    # ── Summary ───────────────────────────────────────────────────────────────
    ok      = [r for r in manifest if r["status"] == "ok"]
    failed  = [r for r in manifest if r["status"] != "ok"]

    log.info("%-20s %s", "Succeeded:", len(ok))
    log.info("%-20s %s", "Failed:", len(failed))

    if failed:
        log.warning("\nFailed sources:")
        for r in failed:
            log.warning("  [%02d] %s  →  %s", r["id"], r["url"], r["error"])

    if ok:
        total_chars = sum(r["raw_text_chars"] or 0 for r in ok)
        log.info("%-20s %s chars across %d docs", "Total raw text:", total_chars, len(ok))

    log.info("=" * 60)
    log.info("Stage 1 complete.  Next: run ingest_and_chunk.py")


if __name__ == "__main__":
    main()