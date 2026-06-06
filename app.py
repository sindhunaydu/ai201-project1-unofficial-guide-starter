"""
app.py — Gradio Interface for the International Student Survival Guide
======================================================================
Architecture position (from planning.md):

  fetch_raw.py        →  raw/
  ingest_and_chunk.py →  output/chunks.json
  embed_store.py      →  chroma_db/
  retrieve.py         →  list[RetrievalResult]
  generate.py         →  GenerationResult
  app.py              →  Gradio UI              ← THIS FILE

UI Structure:
  ┌─────────────────────────────────────────────────┐
  │  Header + status bar                            │
  ├─────────────────────────────┬───────────────────┤
  │  Chat panel (left, 65%)     │  Sources panel    │
  │  • Chatbot history          │  (right, 35%)     │
  │  • Query input + submit     │  • Source cards   │
  │  • Example questions        │  • Debug toggle   │
  └─────────────────────────────┴───────────────────┘

Run:
    python app.py
    python app.py --share          # public Gradio link
    python app.py --port 8080
"""

import argparse
import logging
import os
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from generate import ask, GenerationResult
from retrieve import retrieve, DEFAULT_TOP_K, DEFAULT_THRESHOLD

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger(__name__)

# ── Example questions (drawn from planning.md eval set) ───────────────────────
EXAMPLE_QUESTIONS = [
    "Can I work off-campus during my first year on an F-1 visa?",
    "Do I need to file taxes if I have no income in the US?",
    "How can I rent an apartment without a US credit history?",
    "What should I do if I feel unsafe on campus at night?",
    "What are the signs of culture shock and how do I cope?",
    "How do I open a bank account as an international student?",
    "What is the difference between CPT and OPT?",
    "What happens if I work without proper employment authorization?",
]

# ── CSS ───────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Layout ─────────────────────────────────────────── */
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
footer { display: none !important; }

/* ── Header ─────────────────────────────────────────── */
#app-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2563a8 100%);
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 16px;
    color: white;
}
#app-header h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; color: white; }
#app-header p  { font-size: 14px; margin: 0; opacity: 0.85; color: white; }

/* ── Status bar ─────────────────────────────────────── */
#status-bar {
    font-size: 12px;
    color: #6b7280;
    padding: 6px 4px;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 12px;
}

/* ── Chat panel ─────────────────────────────────────── */
#chatbot { border-radius: 10px; }
#chatbot .message.user   { background: #eff6ff; }
#chatbot .message.bot    { background: #f9fafb; }

/* ── Input row ──────────────────────────────────────── */
#query-row { gap: 8px; }
#query-input textarea { border-radius: 8px; font-size: 15px; }
#submit-btn {
    background: #2563a8 !important;
    color: white !important;
    border-radius: 8px !important;
    min-width: 90px;
}
#submit-btn:hover { background: #1e4f8c !important; }
#clear-btn { border-radius: 8px !important; min-width: 70px; }

/* ── Example pills ──────────────────────────────────── */
.example-pill button {
    background: #f0f4ff !important;
    border: 1px solid #c7d7f8 !important;
    border-radius: 20px !important;
    color: #2563a8 !important;
    font-size: 13px !important;
    padding: 4px 12px !important;
    margin: 3px !important;
}
.example-pill button:hover { background: #dbeafe !important; }

/* ── Source cards ───────────────────────────────────── */
#sources-panel .source-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
#sources-panel { min-height: 200px; }

/* ── Debug panel ────────────────────────────────────── */
#debug-accordion { font-size: 12px; }
#debug-text textarea { font-family: monospace; font-size: 11px; }

/* ── Metric pills ───────────────────────────────────── */
.metric-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
.metric-pill {
    background: #f1f5f9;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    color: #475569;
}
"""

# ── Source card HTML renderer ─────────────────────────────────────────────────

def _render_source_cards(sources: list[dict], chunks_used: int,
                          input_tokens: int, output_tokens: int,
                          elapsed: float) -> str:
    if not sources:
        return "<p style='color:#9ca3af;font-size:14px;padding:12px'>No sources retrieved.</p>"

    seen_urls: set[str] = set()
    cards = []
    for i, s in enumerate(sources, 1):
        url  = s.get("url", "#")
        name = s.get("name", "Unknown source")
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Trim URL for display
        display_url = url.replace("https://", "").replace("http://", "")
        if len(display_url) > 55:
            display_url = display_url[:52] + "…"

        cards.append(f"""
<div style="background:white;border:1px solid #e2e8f0;border-radius:8px;
            padding:11px 14px;margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
    <span style="background:#eff6ff;color:#2563a8;border-radius:50%;
                 width:22px;height:22px;display:inline-flex;align-items:center;
                 justify-content:center;font-size:11px;font-weight:600;
                 flex-shrink:0;">{i}</span>
    <span style="font-weight:500;font-size:13px;color:#1e293b;">{name}</span>
  </div>
  <a href="{url}" target="_blank"
     style="font-size:11px;color:#2563a8;word-break:break-all;">{display_url}</a>
</div>
""")

    metrics = f"""
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;
            padding-top:10px;border-top:1px solid #f1f5f9;">
  <span style="background:#f1f5f9;border-radius:6px;padding:3px 10px;
               font-size:11px;color:#475569;">⏱ {elapsed:.1f}s</span>
  <span style="background:#f1f5f9;border-radius:6px;padding:3px 10px;
               font-size:11px;color:#475569;">📄 {chunks_used} chunks</span>
  <span style="background:#f1f5f9;border-radius:6px;padding:3px 10px;
               font-size:11px;color:#475569;">🔤 {input_tokens}→{output_tokens} tokens</span>
</div>
"""
    return "".join(cards) + metrics


# ── Core chat handler ─────────────────────────────────────────────────────────

def _chat(
    query:       str,
    history:     list[list],
    top_k:       int,
    threshold:   float,
) -> tuple[list, str, str]:
    """
    Called on every user submission.
    Returns (updated_history, sources_html, debug_text).
    """
    query = query.strip()
    if not query:
        return history, "<p style='color:#9ca3af;font-size:14px'>Ask a question to see sources.</p>", ""

    # ── Guard: check GROQ_API_KEY before calling ──────────────────────────
    if not os.getenv("GROQ_API_KEY"):
        err = "⚠️ GROQ_API_KEY is not set. Add it to your `.env` file and restart the app."
        history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": err}]
        return history, "", err

    t0 = time.perf_counter()

    try:
        result: GenerationResult = ask(
            query,
            top_k     = int(top_k),
            threshold = float(threshold),
        )
    except Exception as exc:
        log.exception("Generation error")
        err_msg = f"⚠️ Error: {exc}"
        history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": err_msg}]
        return history, "", err_msg

    elapsed = time.perf_counter() - t0

    history = history + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": result.answer},
    ]

    # Build sources HTML
    sources_html = _render_source_cards(
        result.sources,
        result.chunks_used,
        result.input_tokens,
        result.output_tokens,
        elapsed,
    )

    # Build debug text
    debug_lines = [
        f"Query       : {result.query}",
        f"Model       : {result.model}",
        f"Chunks used : {result.chunks_used}",
        f"Tokens in   : {result.input_tokens}",
        f"Tokens out  : {result.output_tokens}",
        f"Elapsed     : {elapsed:.2f}s",
        "",
        "── Raw model output ──",
        result.raw_response,
    ]
    debug_text = "\n".join(debug_lines)

    return history, sources_html, debug_text


# ── Gradio app ────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:

    with gr.Blocks(title="International Student Survival Guide") as app:

        # ── Header ──────────────────────────────────────────────────────
        gr.HTML("""
<div id="app-header">
  <h1>🎓 International Student Survival Guide</h1>
  <p>Ask anything about F-1 status, OPT/CPT, taxes, housing, banking,
     campus safety, or adjusting to life in the US.</p>
</div>
""")

        # ── Status bar ───────────────────────────────────────────────────
        status_html = gr.HTML(
            value='<div id="status-bar">🟢 Ready — powered by '
                  'all-MiniLM-L6-v2 embeddings · ChromaDB · Groq llama-3.3-70b</div>',
            elem_id="status-bar",
        )

        # ── Main layout: chat (left) + sources (right) ───────────────────
        with gr.Row(equal_height=False):

            # ── Left column — chat ───────────────────────────────────────
            with gr.Column(scale=6):

                chatbot = gr.Chatbot(
                    label      = "Conversation",
                    elem_id    = "chatbot",
                    height     = 460,
                    show_label = True,
                )

                with gr.Row(elem_id="query-row"):
                    query_input = gr.Textbox(
                        placeholder = "e.g. How do I apply for OPT after graduation?",
                        show_label  = False,
                        lines       = 2,
                        max_lines   = 4,
                        elem_id     = "query-input",
                        scale       = 5,
                    )
                    with gr.Column(scale=1, min_width=100):
                        submit_btn = gr.Button("Ask ↗",  variant="primary",
                                               elem_id="submit-btn")
                        clear_btn  = gr.Button("Clear",  variant="secondary",
                                               elem_id="clear-btn")

                # ── Example questions ────────────────────────────────────
                gr.Markdown("**Try an example:**", elem_classes=["example-label"])
                gr.Examples(
                    examples   = [[q] for q in EXAMPLE_QUESTIONS],
                    inputs     = [query_input],
                    label      = "",
                    elem_id    = "examples",
                    examples_per_page = 4,
                )

                # ── Advanced settings (collapsed by default) ─────────────
                with gr.Accordion("⚙️ Retrieval settings", open=False):
                    with gr.Row():
                        top_k_slider = gr.Slider(
                            minimum=4, maximum=6, step=1, value=DEFAULT_TOP_K,
                            label="Top-k chunks",
                            info="Number of context chunks to retrieve (4–6)",
                        )
                        threshold_slider = gr.Slider(
                            minimum=0.1, maximum=0.9, step=0.05,
                            value=DEFAULT_THRESHOLD,
                            label="Similarity threshold",
                            info="Minimum cosine similarity to include a chunk",
                        )

            # ── Right column — sources + debug ───────────────────────────
            with gr.Column(scale=4):

                gr.Markdown("### Sources")
                sources_panel = gr.HTML(
                    value   = "<p style='color:#9ca3af;font-size:14px;"
                              "padding:12px'>Ask a question to see sources.</p>",
                    elem_id = "sources-panel",
                )

                with gr.Accordion("🔍 Debug / raw output", open=False,
                                  elem_id="debug-accordion"):
                    debug_box = gr.Textbox(
                        label     = "",
                        lines     = 12,
                        max_lines = 20,
                        interactive = False,
                        elem_id   = "debug-text",
                    )

        # ── State — Gradio messages format: list of {"role", "content"} dicts ──
        history_state = gr.State([])

        # ── Event wiring ─────────────────────────────────────────────────

        def _handle_submit(query, history, top_k, threshold):
            new_history, sources_html, debug_text = _chat(
                query, history, top_k, threshold
            )
            return new_history, new_history, sources_html, debug_text, ""

        def _handle_clear():
            return [], [], "<p style='color:#9ca3af;font-size:14px;padding:12px'>Ask a question to see sources.</p>", ""

        submit_inputs  = [query_input, history_state, top_k_slider, threshold_slider]
        submit_outputs = [chatbot, history_state, sources_panel, debug_box, query_input]

        submit_btn.click(
            fn      = _handle_submit,
            inputs  = submit_inputs,
            outputs = submit_outputs,
        )
        query_input.submit(
            fn      = _handle_submit,
            inputs  = submit_inputs,
            outputs = submit_outputs,
        )
        clear_btn.click(
            fn      = _handle_clear,
            outputs = [chatbot, history_state, sources_panel, debug_box],
        )

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int,  default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    parser.add_argument("--debug", action="store_true",
                        help="Enable Gradio debug mode")
    args = parser.parse_args()

    # Warn early if GROQ_API_KEY is missing
    if not os.getenv("GROQ_API_KEY"):
        log.warning(
            "GROQ_API_KEY is not set. The app will start but queries will fail.\n"
            "Create a .env file with:  GROQ_API_KEY=gsk_..."
        )

    app = build_app()
    app.launch(
        server_port = args.port,
        share       = args.share,
        debug       = args.debug,
        show_error  = True,
        css         = CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
