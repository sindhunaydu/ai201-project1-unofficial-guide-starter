"""
generate.py — Stage 5: Grounded Answer Generation via Groq
===========================================================
Architecture position (from planning.md):

  retrieve.py  →  list[RetrievalResult]
  generate.py  →  GenerationResult          ← THIS FILE
  app.py       →  Gradio UI

What this module does:
  1. Accepts a user query + retrieved chunks from retrieve.py
  2. Builds a grounded system prompt — model must answer ONLY from context
  3. Calls the Groq API (llama-3.3-70b-versatile) for fast, cheap inference
  4. Returns a structured GenerationResult with answer text + cited sources

Grounding policy (enforced in system prompt):
  - Every claim must trace to a retrieved chunk
  - If the context does not contain an answer, say so explicitly
  - Sources are listed at the end in a fixed format the UI can parse
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

from retrieve import RetrievalResult, retrieve

load_dotenv()
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_MODEL   = "llama-3.3-70b-versatile"
MAX_TOKENS   = 1024
TEMPERATURE  = 0.2     # low: factual grounded answers, minimal hallucination
DEFAULT_TOP_K = 5

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the International Student Survival Guide — a reliable, friendly assistant \
that helps international students navigate life in the United States.

GROUNDING RULES (follow exactly):
1. Answer ONLY using information present in the CONTEXT CHUNKS below.
2. If the context does not contain enough information to answer the question, \
respond with:
   "I don't have enough information in my sources to answer that. \
Please check with your university's international student office or \
the relevant government agency."
3. Do NOT invent facts, dates, form numbers, deadlines, or dollar amounts \
that are not explicitly stated in the context.
4. Do NOT answer from general knowledge if it contradicts or extends the context.

OUTPUT FORMAT (follow exactly):
Provide your response in two clearly separated sections:

### Answer
[Your answer here. Use plain paragraphs. You may use a short bulleted list if \
listing 3+ items. Keep the answer concise — 150 to 300 words.]

### Sources
[List every source you drew on, one per line, in this exact format:]
- Source Name | URL

STYLE:
- Friendly and clear — you are writing for students, not lawyers.
- Use plain language. Spell out abbreviations on first use.
- If a deadline or form number appears in the context, include it.
- End with one actionable next step if the context supports it.
"""


def _build_context_block(chunks: list[RetrievalResult]) -> str:
    """
    Format retrieved chunks into the CONTEXT CHUNKS section of the user message.
    Each chunk is labelled with its rank and source so the model can attribute.
    """
    parts = []
    for r in chunks:
        parts.append(
            f"[Chunk {r.rank} | {r.source_name} | score={r.score:.3f}]\n"
            f"{r.text}"
        )
    return "\n\n---\n\n".join(parts)


def _parse_sources(answer_text: str) -> tuple[str, list[dict]]:
    """
    Split the model's response into (answer_body, sources_list).
    Parses lines under '### Sources' that match '- Name | URL'.
    Returns the answer section (without the sources block) and a list of
    {'name': ..., 'url': ...} dicts.
    """
    sources: list[dict] = []

    # Split on the Sources heading (case-insensitive, optional whitespace)
    parts = re.split(r"(?im)^###\s*sources\s*$", answer_text, maxsplit=1)

    answer_body = parts[0].strip()

    if len(parts) == 2:
        for line in parts[1].strip().splitlines():
            line = line.strip()
            if line.startswith("-"):
                line = line[1:].strip()
            if "|" in line:
                name, _, url = line.partition("|")
                sources.append({"name": name.strip(), "url": url.strip()})

    # Remove the '### Answer' heading if present
    answer_body = re.sub(r"(?im)^###\s*answer\s*$", "", answer_body).strip()

    return answer_body, sources


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    query:          str
    answer:         str                  # clean answer text (no sources block)
    sources:        list[dict]           # [{"name": ..., "url": ...}, ...]
    chunks_used:    int
    model:          str
    input_tokens:   int
    output_tokens:  int
    raw_response:   str                  # full model output for debugging


# ── Core generation function ──────────────────────────────────────────────────

def generate(
    query:  str,
    chunks: list[RetrievalResult],
) -> GenerationResult:
    """
    Generate a grounded answer for `query` using `chunks` as context.

    Parameters
    ----------
    query  : The user's question.
    chunks : Retrieved chunks from retrieve.py — passed in so the caller
             controls retrieval parameters (top_k, threshold, tag_filter).

    Returns
    -------
    GenerationResult with parsed answer, source list, and token usage.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file:\n"
            "  GROQ_API_KEY=gsk_..."
        )

    client = Groq(api_key=api_key)

    if not chunks:
        # No context retrieved — return a grounded refusal rather than guessing
        no_context_answer = (
            "I don't have enough information in my sources to answer that. "
            "Please check with your university's international student office "
            "or the relevant government agency."
        )
        return GenerationResult(
            query         = query,
            answer        = no_context_answer,
            sources       = [],
            chunks_used   = 0,
            model         = GROQ_MODEL,
            input_tokens  = 0,
            output_tokens = 0,
            raw_response  = no_context_answer,
        )

    context_block = _build_context_block(chunks)

    user_message = (
        f"CONTEXT CHUNKS:\n\n{context_block}\n\n"
        f"QUESTION: {query}"
    )

    log.info(
        "Calling Groq (%s) — %d chunks, query: %s",
        GROQ_MODEL, len(chunks), query[:80],
    )

    response = client.chat.completions.create(
        model       = GROQ_MODEL,
        temperature = TEMPERATURE,
        max_tokens  = MAX_TOKENS,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    raw = response.choices[0].message.content or ""
    answer_body, sources = _parse_sources(raw)

    usage = response.usage
    log.info(
        "Generation complete — in=%d out=%d tokens",
        usage.prompt_tokens, usage.completion_tokens,
    )

    return GenerationResult(
        query         = query,
        answer        = answer_body,
        sources       = sources,
        chunks_used   = len(chunks),
        model         = GROQ_MODEL,
        input_tokens  = usage.prompt_tokens,
        output_tokens = usage.completion_tokens,
        raw_response  = raw,
    )


def ask(
    query:      str,
    top_k:      int   = DEFAULT_TOP_K,
    threshold:  float = 0.30,
    tag_filter: Optional[str] = None,
) -> GenerationResult:
    """
    One-call convenience: retrieve + generate.
    Used by app.py and CLI.
    """
    chunks = retrieve(query, top_k=top_k, threshold=threshold, tag_filter=tag_filter)
    return generate(query, chunks)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ask the International Student Guide")
    parser.add_argument("query", nargs="?",
                        default="What is OPT and how do I apply for it?",
                        help="Question to answer")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--tag",   type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")

    result = ask(args.query, top_k=args.top_k, tag_filter=args.tag)

    print(f"\n{'='*68}")
    print(f"Q: {result.query}")
    print(f"{'='*68}")
    print(result.answer)
    print(f"\n{'─'*68}")
    print("Sources:")
    for s in result.sources:
        print(f"  • {s['name']}")
        print(f"    {s['url']}")
    print(f"\nTokens: {result.input_tokens} in / {result.output_tokens} out"
          f" | chunks used: {result.chunks_used}")