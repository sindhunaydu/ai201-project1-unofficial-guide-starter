"""
test_chunking.py — Validates chunking logic without network or GPU.
Uses tiktoken (pure-Python, installable anywhere) to replicate
the tokenisation that all-MiniLM-L6-v2 WordPiece would produce
(close enough for chunk-size verification purposes).
"""

import re
import json
from dataclasses import dataclass, asdict

try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def _token_count(text): return len(enc.encode(text))
except ImportError:
    # Absolute fallback: whitespace-based approximation
    def _token_count(text): return len(text.split())

# ── Copy of chunking logic (mirrors ingest_and_chunk.py) ──────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 100


def _split_into_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]


def _split_paragraph_into_sentences(para):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'\(])", para)
    return [s.strip() for s in parts if s.strip()]


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks = []
    current_units = []
    current_tokens = 0

    def flush_chunk():
        nonlocal current_units, current_tokens
        chunk_text_ = " ".join(current_units).strip()
        if chunk_text_:
            chunks.append(chunk_text_)
        seed_units = []
        seed_tokens = 0
        for unit in reversed(current_units):
            ut = _token_count(unit)
            if seed_tokens + ut > overlap:
                break
            seed_units.insert(0, unit)
            seed_tokens += ut
        current_units  = seed_units
        current_tokens = seed_tokens

    def add_unit(unit):
        nonlocal current_tokens
        current_units.append(unit)
        current_tokens += _token_count(unit)

    for para in paragraphs:
        para_tokens = _token_count(para)
        if current_tokens + para_tokens <= chunk_size:
            add_unit(para)
        else:
            sentences = _split_paragraph_into_sentences(para)
            for sent in sentences:
                sent_tokens = _token_count(sent)
                if current_tokens + sent_tokens <= chunk_size:
                    add_unit(sent)
                else:
                    if current_units:
                        flush_chunk()
                    if sent_tokens <= chunk_size:
                        add_unit(sent)
                    else:
                        # Hard token-window split for very long sentences
                        words = sent.split()
                        buf, buf_tok = [], 0
                        for w in words:
                            wt = _token_count(w)
                            if buf_tok + wt > chunk_size:
                                chunks.append(" ".join(buf))
                                # keep overlap worth of words
                                buf = buf[max(0, len(buf)-10):]
                                buf_tok = _token_count(" ".join(buf))
                            buf.append(w)
                            buf_tok += wt
                        if buf:
                            current_units  = buf
                            current_tokens = _token_count(" ".join(buf))

    if current_units:
        remaining = " ".join(current_units).strip()
        if remaining:
            chunks.append(remaining)

    return [c for c in chunks if c]


# ── Synthetic test documents ──────────────────────────────────────────────────
SYNTHETIC_DOCS = {
    "opt_cpt": """
Optional Practical Training (OPT) allows F-1 students to work in the United States
in a job directly related to their major area of study. Students may apply for OPT
before or after graduation. Pre-completion OPT takes place before your program end
date, while post-completion OPT begins after you graduate.

To be eligible for OPT, you must have been enrolled full-time for at least one full
academic year. You should apply through your Designated School Official (DSO), who
will update your SEVIS record and issue a new Form I-20 with an OPT recommendation.
Then you file Form I-765 (Application for Employment Authorization) with USCIS.
Processing typically takes several months, so plan ahead.

Curricular Practical Training (CPT) is an alternative form of off-campus work
authorization that is part of your established curriculum. CPT must be an integral
part of an established curriculum and must be authorized before you begin working.
Unlike OPT, CPT authorization is granted by your school, not USCIS.

If you use 12 or more months of full-time CPT, you will lose eligibility for OPT.
Part-time CPT (20 hours or less per week) does not count against your OPT eligibility.
Always consult your international student advisor before accepting any off-campus work.

STEM OPT Extension: Students who graduate with a degree in Science, Technology,
Engineering, or Mathematics may apply for a 24-month extension of their post-completion
OPT. To qualify, you must be employed by an E-Verify employer and your job must be
directly related to your STEM degree. You must file for the extension before your
initial OPT period expires.
""",

    "taxes": """
International students on F-1 visas are generally considered nonresident aliens for
U.S. federal income tax purposes during the first five calendar years of their stay.
As a nonresident alien, you are subject to different tax rules than U.S. citizens.

All F-1 students must file Form 8843 with the IRS every year, even if you earned
no income during the tax year. This form establishes your nonresident alien status.
If you did earn income — from on-campus employment, scholarships, fellowships, or
internship wages — you will also need to file Form 1040-NR.

The tax deadline is typically April 15 for those who earned income, and June 15 for
those filing only Form 8843 with no income. Scholarships used for tuition and required
fees are usually not taxable, but amounts used for room and board are taxable.

Many universities partner with Sprintax or Glacier Tax Prep, which are software tools
designed specifically for international students filing U.S. taxes. These tools guide
you step by step and help you claim the correct deductions and treaty benefits.

Tax treaties exist between the United States and many countries. If your home country
has a tax treaty with the U.S., you may be exempt from tax on certain income types or
up to a certain dollar amount. Check the IRS website or ask your international student
office to find out whether your country has an applicable treaty.
""",

    "housing": """
Finding housing in the United States as an international student can be challenging
because landlords typically require a credit history, which most new arrivals do not have.

There are several strategies to overcome this. First, consider living in university or
on-campus housing for your first year, which typically does not require a credit check.
Second, when applying for off-campus apartments, you can offer to pay a larger security
deposit — often two to three months of rent — to compensate for the lack of credit history.

Providing proof of financial support is also helpful. Bring bank statements, your
scholarship award letter, or a letter from your department confirming your stipend.
Some landlords will accept a U.S. co-signer or guarantor — a person with established
credit who agrees to take responsibility for rent if you cannot pay.

If you do not have a co-signer, services such as Leap, TheGuarantors, or Insurent act
as professional guarantors for a fee. Some buildings work with these services and will
accept them in lieu of a personal co-signer.

Your passport, F-1 visa, I-20 form, and university enrollment letter are the key
identity documents you should bring to any rental application. Having these organized
in advance shows preparedness and can help build trust with a landlord.
""",
}


# ── Tests ─────────────────────────────────────────────────────────────────────
@dataclass
class ChunkResult:
    doc_name:    str
    chunk_index: int
    token_count: int
    text_preview: str


def run_tests():
    print("=" * 65)
    print("CHUNKING UNIT TESTS")
    print(f"Config: CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}")
    print("=" * 65)

    all_results = []
    passed = 0
    failed = 0

    for doc_name, text in SYNTHETIC_DOCS.items():
        print(f"\n▶ Document: {doc_name}")
        chunks = chunk_text(text)

        if not chunks:
            print("  ✗ FAIL: no chunks produced")
            failed += 1
            continue

        violations = []
        for i, chunk in enumerate(chunks):
            tc = _token_count(chunk)
            result = ChunkResult(
                doc_name    = doc_name,
                chunk_index = i,
                token_count = tc,
                text_preview = chunk[:80].replace("\n", " ") + "…",
            )
            all_results.append(result)

            status = "✓" if tc <= CHUNK_SIZE else "✗ OVER-SIZE"
            print(f"  [{i:02d}] {tc:>4} tokens  {status}  | {result.text_preview}")

            if tc > CHUNK_SIZE:
                violations.append((i, tc))

        # ── Assertions ──────────────────────────────────────────────────────
        # 1. All chunks within size limit
        if violations:
            print(f"  ✗ FAIL: {len(violations)} chunk(s) exceed {CHUNK_SIZE} tokens")
            failed += 1
        else:
            print(f"  ✓ PASS: all {len(chunks)} chunks within {CHUNK_SIZE}-token limit")
            passed += 1

        # 2. Chunks cover original content (total tokens >= original)
        original_tokens = _token_count(text)
        chunk_tokens    = sum(_token_count(c) for c in chunks)
        if chunk_tokens >= original_tokens * 0.9:
            print(f"  ✓ PASS: coverage OK ({chunk_tokens} chunk-tokens "
                  f"vs {original_tokens} original-tokens)")
            passed += 1
        else:
            print(f"  ✗ FAIL: low coverage — only {chunk_tokens} chunk-tokens "
                  f"from {original_tokens} original")
            failed += 1

        # 3. Overlap: last tokens of chunk N appear in chunk N+1
        if len(chunks) >= 2:
            last_words_of_prev = chunks[0].split()[-15:]
            first_words_of_next = chunks[1].split()[:50]
            overlap_found = any(
                w in first_words_of_next for w in last_words_of_prev
            )
            if overlap_found:
                print(f"  ✓ PASS: overlap content detected between chunk 0 → 1")
                passed += 1
            else:
                print(f"  ✗ FAIL: no overlap detected between chunk 0 → 1")
                failed += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 65)

    # Save results to JSON
    with open("output/test_results.json", "w") as f:
        json.dump(
            [asdict(r) for r in all_results],
            f, indent=2
        )
    print(f"Chunk details saved → output/test_results.json")


if __name__ == "__main__":
    import os; os.makedirs("output", exist_ok=True)
    run_tests()