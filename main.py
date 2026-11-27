import os
import re
import sys
import tempfile
import threading
import webbrowser

# ---------------------------------------------------------------------------
# Terminal commands (UI ke bina) - heavy imports se pehle handle hote hain
# ---------------------------------------------------------------------------
if len(sys.argv) > 1 and sys.argv[1] in ("ingest", "clear"):
    if sys.argv[1] == "ingest":
        from ingest import ingest, DEFAULT_PDF

        ingest(sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PDF)
    else:
        from clear_index import clear

        clear()
    sys.exit(0)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

from data_loader import PDFLoader
from models import LLM
from prompt import PROMPT
from retriever import doc_out
from splitter import SPLITTER
from vector_db import VECTOR_DB
import vector_db

app = FastAPI(title="Glean")

file_path = os.getenv("PDF_PATH", "/app/attention_is_all_you_need.pdf")

pdf = PDFLoader(file_path)
documents = pdf.load_pdf()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = VECTOR_DB(documents, embeddings)
db.vector_store = db.create_vector_store()

print("VECTOR DB FILE:", vector_db.__file__)
print("CLEAR METHOD:", hasattr(db, "clear"))
print("ADD DOCUMENTS METHOD:", hasattr(db, "add_documents"))


answer_chain = PROMPT().get_prompt() | LLM | StrOutputParser()


def _resolve_index_name(vector_db) -> str:
    for attr in ("index_name", "collection_name", "name"):
        val = getattr(vector_db, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "default"


def _safe_clear():
    """Index pehle se khali ho to Pinecone 404 (Namespace not found) deta hai - use ignore karo."""
    try:
        db.clear()
    except Exception as e:
        msg = str(e)
        if "Namespace not found" in msg or "404" in msg:
            return
        raise


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=8)


@app.get("/api/status")
def status():
    count = None
    index_name = _resolve_index_name(db)
    try:
        pc = getattr(db, "pc", None)
        if pc is not None and index_name != "default":
            stats = pc.Index(index_name).describe_index_stats()
            count = getattr(stats, "total_vector_count", None)
            if count is None and isinstance(stats, dict):
                count = stats.get("total_vector_count")
        elif hasattr(db, "vector_store") and hasattr(db.vector_store, "_collection"):
            count = db.vector_store._collection.count()
    except Exception:
        pass
    return {"index": index_name, "vectors": count}


@app.post("/api/ask")
def ask(body: AskIn):
    try:
        docs = db.vector_store.as_retriever(search_kwargs={"k": body.k}).invoke(
            body.question
        )
        answer = answer_chain.invoke(
            {"context": doc_out(docs), "question": body.question}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not answer: {e}")

    sources = []
    for i, d in enumerate(docs):
        page = d.metadata.get("page")
        sources.append(
            {
                "n": i + 1,
                "page": int(page) + 1 if page is not None else None,
                "text": d.page_content.strip(),
            }
        )
    return {"answer": answer, "sources": sources}


@app.post("/api/upload")
def upload(file: UploadFile = File(...), replace: bool = Form(True)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.file.read())
        path = tmp.name
    try:
        pages = PDFLoader(path).load_pdf()
        chunks = SPLITTER().split_documents(pages)
        if not chunks:
            raise HTTPException(
                status_code=422, detail="No text found. Is this a scanned PDF?"
            )
        if replace:
            _safe_clear()
        prefix = re.sub(r"[^A-Za-z0-9_-]", "_", os.path.splitext(file.filename)[0])
        db.add_documents(chunks, prefix=prefix)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    finally:
        if os.path.exists(path):
            os.remove(path)

    return {"filename": file.filename, "pages": len(pages), "chunks": len(chunks)}


@app.post("/api/clear")
def clear_index():
    _safe_clear()
    return {"ok": True}


# ---------------------------------------------------------------------------
# UI (React + CSS, ek hi page)
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
<meta name="theme-color" content="#3347c9" />
<meta name="mobile-web-app-capable" content="yes" />
<title>Glean</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap" rel="stylesheet" />
<script>
(function () {
  var t = null;
  try { t = localStorage.getItem('marginalia-theme'); } catch (e) {}
  if (!t) t = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', t);
})();
</script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" />
<style>
:root {
  --bg: #eef1f6; --surface: #ffffff; --surface-2: #f6f8fb; --ink: #151c2b; --muted: #5d6a80; --line: #dbe0ea;
  --accent: #3347c9; --accent-ink: #ffffff; --accent-soft: #e8ebfb; --mark: #ffe27a; --mark-ink: #3a2f00;
  --danger: #b3261e; --ok: #1d7a46; --shadow: 0 1px 2px rgba(20,30,60,.06), 0 6px 20px rgba(20,30,60,.06);
  --serif: 'Newsreader', Georgia, 'Times New Roman', serif;
  --sans: 'Instrument Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
}
:root[data-theme="dark"] {
  --bg: #0d121c; --surface: #151c2a; --surface-2: #1a2334; --ink: #e7ecf5; --muted: #93a0b7; --line: #263149;
  --accent: #8ea2ff; --accent-ink: #0d121c; --accent-soft: #212c4d; --mark: #5b4a0e; --mark-ink: #fff1b8;
  --danger: #ff8a80; --ok: #6fd39a; --shadow: 0 1px 2px rgba(0,0,0,.4), 0 6px 20px rgba(0,0,0,.3);
}
*, *::before, *::after { box-sizing: border-box; }
html, body, #root { height: 100%; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.5; -webkit-font-smoothing: antialiased; }
button, input, textarea { font: inherit; color: inherit; }
button { cursor: pointer; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
code { font-size: .88em; background: var(--surface-2); border: 1px solid var(--line); padding: 1px 5px; border-radius: 4px; }
::-webkit-scrollbar { width: 10px; } ::-webkit-scrollbar-thumb { background: var(--line); border-radius: 8px; border: 2px solid transparent; background-clip: content-box; }

.app { display: grid; grid-template-columns: 290px minmax(0, 1fr) 380px; height: 100dvh; }
.conversation { display: flex; flex-direction: column; min-width: 0; height: 100dvh; }
.topbar, .scrim { display: none; }

.btn { border: 0; border-radius: 9px; padding: 9px 16px; background: var(--accent); color: var(--accent-ink); font-weight: 600; transition: filter .15s, border-color .15s; }
.btn:hover:not(:disabled) { filter: brightness(1.08); }
.btn:disabled { opacity: .45; cursor: not-allowed; }
.btn.ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); font-weight: 500; }
.btn.ghost:hover:not(:disabled) { border-color: var(--accent); filter: none; }
.btn.danger { color: var(--danger); }
.btn.sm { padding: 6px 12px; font-size: 14px; }
.row { display: flex; gap: 8px; } .row .btn { flex: 1; }
.link { background: none; border: 0; padding: 0; color: var(--accent); font-size: 14px; font-weight: 500; text-decoration: underline; text-underline-offset: 3px; }
.close { display: none; margin-left: auto; width: 32px; height: 32px; border-radius: 8px; border: 1px solid var(--line); background: transparent; font-size: 20px; line-height: 1; }

.sidebar, .evidence { background: var(--surface); overflow-y: auto; display: flex; flex-direction: column; }
.sidebar { border-right: 1px solid var(--line); padding: 22px 20px; gap: 22px; }
.brand { display: flex; align-items: center; gap: 10px; font-family: var(--serif); font-size: 24px; font-weight: 600; letter-spacing: -.01em; }
.brand .mark-fill { fill: var(--mark); }
.side-title { margin: 0; font-size: 13.5px; font-weight: 600; color: var(--muted); }
.sidebar section { display: flex; flex-direction: column; gap: 10px; }
.push-down { margin-top: auto; }
.drop { display: flex; flex-direction: column; align-items: center; gap: 2px; text-align: center; padding: 20px 12px; border: 1.5px dashed var(--line); border-radius: 12px; cursor: pointer; overflow-wrap: anywhere; position: relative; background: var(--surface-2); transition: border-color .15s, background-color .15s; }
.drop:hover, .drop:focus-within, .drop.over { border-color: var(--accent); background: var(--accent-soft); }
.drop input { position: absolute; width: 1px; height: 1px; opacity: 0; }
.drop span { color: var(--muted); font-size: 13px; }
.drop.has-file { border-style: solid; border-color: var(--accent); }
.check { display: flex; align-items: center; gap: 8px; font-size: 14px; }
.check input { accent-color: var(--accent); width: 16px; height: 16px; }
.range { display: flex; flex-direction: column; gap: 6px; font-size: 14px; }
.range b { margin-left: 4px; font-variant-numeric: tabular-nums; }
.range input { accent-color: var(--accent); }
.hint { margin: 0; font-size: 12.5px; color: var(--muted); }
.stats { margin: 0; display: grid; gap: 8px; padding: 12px 14px; background: var(--surface-2); border: 1px solid var(--line); border-radius: 10px; }
.stats div { display: flex; justify-content: space-between; font-size: 14px; }
.stats dt { color: var(--muted); }
.stats dd { margin: 0; font-weight: 600; font-variant-numeric: tabular-nums; display: flex; align-items: center; gap: 6px; }
.stats dd i { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.stats dd.on { color: var(--ok); } .stats dd.off { color: var(--danger); }
.notice { margin: 0; font-size: 13.5px; padding: 9px 11px; border-radius: 8px; border: 1px solid var(--line); }
.notice.ok { border-color: var(--ok); color: var(--ok); }
.notice.bad { border-color: var(--danger); color: var(--danger); }
.notice .link { margin-left: 8px; }

.thread { flex: 1; overflow-y: auto; padding: 36px 28px 12px; }
.thread-inner { max-width: 720px; margin: 0 auto; }
.empty { margin-top: 8vh; display: flex; flex-direction: column; gap: 14px; }
.empty h1 { margin: 0; font-family: var(--serif); font-weight: 500; font-size: 38px; line-height: 1.15; letter-spacing: -.015em; }
.empty > p { margin: 0; color: var(--muted); font-size: 16px; }
.suggest { list-style: none; padding: 0; margin: 14px 0 0; display: flex; flex-direction: column; gap: 8px; }
.suggest button { width: 100%; text-align: left; padding: 12px 15px; border-radius: 10px; background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow); transition: border-color .15s; }
.suggest button:hover:not(:disabled) { border-color: var(--accent); }
.turn { margin-bottom: 36px; }
.question { margin: 0 0 12px; font-size: 16px; font-weight: 600; }
.answer { font-family: var(--serif); font-size: 19px; line-height: 1.65; border-left: 3px solid var(--line); padding-left: 18px; }
.turn.is-active .answer { border-left-color: var(--accent); }
.answer p { margin: 0 0 .8em; }
.answer p:last-child, .answer ul:last-child { margin-bottom: 0; }
.answer ul { margin: 0 0 .8em; padding-left: 1.2em; }
.answer li { margin-bottom: .3em; }
.actions { display: flex; align-items: center; gap: 14px; margin: 12px 0 0 21px; flex-wrap: wrap; }
.actions .link { color: var(--muted); text-decoration: none; padding: 2px 0; }
.actions .link:hover { color: var(--accent); text-decoration: underline; }
.actions .link.main { color: var(--accent); }
.cite { display: inline-grid; place-items: center; min-width: 20px; height: 20px; padding: 0 6px; margin: 0 2px; border-radius: 999px; border: 0; background: var(--accent); color: var(--accent-ink); font: 600 12px var(--sans); vertical-align: 2px; }
.cite:hover { filter: brightness(1.12); }
.typing { display: flex; gap: 6px; padding: 8px 0 8px 21px; }
.typing i { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); animation: pulse 1.1s infinite ease-in-out; }
.typing i:nth-child(2) { animation-delay: .15s; } .typing i:nth-child(3) { animation-delay: .3s; }
@keyframes pulse { 0%, 80%, 100% { opacity: .25; } 40% { opacity: 1; } }

.composer-wrap { padding: 12px 28px 20px; }
.composer { max-width: 720px; margin: 0 auto; display: flex; gap: 10px; align-items: flex-end; background: var(--surface); border: 1px solid var(--line); border-radius: 14px; padding: 8px 8px 8px 16px; box-shadow: var(--shadow); }
.composer:focus-within { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 22%, transparent); }
.composer textarea { flex: 1; border: 0; outline: 0; resize: none; background: transparent; max-height: 140px; line-height: 1.5; padding: 7px 0; }
.composer-meta { max-width: 720px; margin: 6px auto 0; display: flex; justify-content: space-between; font-size: 12.5px; color: var(--muted); padding: 0 4px; }
.composer-meta .over { color: var(--danger); }

.evidence { border-left: 1px solid var(--line); padding: 22px 20px; }
.evidence-head { display: flex; align-items: center; margin-bottom: 8px; }
.evidence h2 { margin: 0; font-family: var(--serif); font-size: 23px; font-weight: 600; }
.muted { color: var(--muted); font-size: 14px; margin: 0 0 14px; }
.ask-ref { overflow-wrap: anywhere; }
.excerpt { border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; margin-bottom: 12px; cursor: pointer; background: var(--surface-2); transition: background-color .2s, border-color .2s; }
.excerpt:hover { border-color: var(--accent); }
.excerpt header { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 12.5px; font-weight: 600; color: var(--muted); }
.excerpt p { margin: 0; font-family: var(--serif); font-size: 15px; line-height: 1.6; overflow-wrap: anywhere; }
.badge { display: inline-grid; place-items: center; min-width: 20px; height: 20px; padding: 0 6px; border-radius: 999px; background: var(--accent); color: var(--accent-ink); font-size: 12px; }
.excerpt.is-on { background: var(--mark); color: var(--mark-ink); border-color: transparent; }
.excerpt.is-on header { color: inherit; }

.toast { position: fixed; left: 50%; bottom: 28px; transform: translateX(-50%); z-index: 40; padding: 10px 16px; border-radius: 10px; background: var(--ink); color: var(--bg); font-size: 14px; font-weight: 500; box-shadow: var(--shadow); max-width: 90vw; }

.answer ol { margin: 0 0 .8em; padding-left: 1.4em; }
.answer h3 { font-family: var(--sans); font-size: 16px; font-weight: 600; margin: 1.1em 0 .4em; }
.answer .katex { font-size: 1.06em; }
.math-block { overflow-x: auto; overflow-y: hidden; padding: .5em 0; margin: .4em 0 .8em; text-align: center; }
.math-block .katex-display { margin: 0; }
.convbar { display: flex; align-items: center; gap: 10px; padding: 8px 20px; background: var(--surface); border-bottom: 1px solid var(--line); }
.conv-title { font-family: var(--serif); font-size: 18px; font-weight: 600; }
.grow { flex: 1; }
.pill { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: var(--ok); }
.pill i { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
.pill.off { color: var(--danger); }
.icon { display: inline-grid; place-items: center; width: 34px; height: 34px; border-radius: 50%; border: 0; background: transparent; color: var(--muted); transition: background-color .15s, color .15s; }
.icon:hover { background: var(--surface-2); color: var(--ink); }
.icon.on { background: var(--accent-soft); color: var(--accent); }
.composer { flex-direction: column; align-items: stretch; gap: 4px; padding: 12px 12px 10px 16px; }
.composer textarea { flex: none; min-height: 28px; }
.tools { display: flex; align-items: center; gap: 4px; }
.doc-chip { max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; color: var(--muted); margin-left: 6px; }
.menu-wrap { position: relative; }
.menu { position: absolute; bottom: 42px; left: 0; min-width: 220px; z-index: 30; display: flex; flex-direction: column; padding: 6px; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); }
.menu button { text-align: left; border: 0; background: transparent; padding: 9px 12px; border-radius: 8px; font-size: 14px; }
.menu button:hover { background: var(--surface-2); }
.lang { border: 0; background: transparent; color: var(--muted); font-size: 12.5px; max-width: 78px; cursor: pointer; }
.send { width: 36px; height: 36px; border-radius: 50%; border: 0; margin-left: 4px; display: inline-grid; place-items: center; background: var(--accent); color: var(--accent-ink); }
.send:disabled { opacity: .4; cursor: not-allowed; }
.composer-meta { justify-content: center; gap: 12px; }
.app.no-side { grid-template-columns: minmax(0, 1fr) 380px; }
.app.no-side.no-evidence { grid-template-columns: minmax(0, 1fr); }
.app.no-side .sidebar { display: none; }
.app.no-evidence { grid-template-columns: 290px minmax(0, 1fr); }
.app.no-evidence .evidence { display: none; }
@media (max-width: 1100px) {
  .convbar { display: none; }
  .app.no-side, .app.no-side.no-evidence { grid-template-columns: minmax(0, 1fr); }
  .app.no-side .sidebar { display: flex; }
  .app.no-evidence { grid-template-columns: minmax(0, 1fr); }
  .app.no-evidence .evidence { display: flex; }
  .app { grid-template-columns: minmax(0, 1fr); }
  .topbar { display: flex; align-items: center; justify-content: space-between; padding: 10px 16px; background: var(--surface); border-bottom: 1px solid var(--line); }
  .brand-sm { font-family: var(--serif); font-size: 20px; font-weight: 600; }
  .topbar-actions { display: flex; gap: 8px; }
  .sidebar, .evidence { position: fixed; top: 0; bottom: 0; width: min(92vw, 380px); z-index: 20; transition: transform .22s ease; box-shadow: 0 0 40px rgba(0,0,0,.25); }
  .sidebar { left: 0; border-right: 0; transform: translateX(-105%); }
  .evidence { right: 0; border-left: 0; transform: translateX(105%); }
  .sidebar.open, .evidence.open { transform: none; }
  .scrim { display: block; position: fixed; inset: 0; z-index: 10; background: rgba(10,15,25,.45); }
  .close { display: inline-grid; place-items: center; }
  .thread { padding: 24px 16px 8px; }
  .composer-wrap { padding: 10px 16px 14px; }
  .empty h1 { font-size: 30px; } .answer { font-size: 18px; }
}
html { -webkit-text-size-adjust: 100%; }
body { overflow-x: hidden; }
@media (max-width: 640px) {
  .topbar { padding: 8px 12px; gap: 8px; }
  .topbar-actions { align-items: center; }
  .btn.sm { padding: 8px 12px; }
  .thread { padding: 18px 14px 8px; }
  .empty { margin-top: 3vh; }
  .empty h1 { font-size: 26px; }
  .empty > p { font-size: 15px; }
  .answer { font-size: 17px; padding-left: 14px; }
  .actions { margin-left: 17px; gap: 12px; }
  .composer-wrap { padding: 8px 12px calc(10px + env(safe-area-inset-bottom, 0px)); }
  .composer { border-radius: 18px; padding: 10px 10px 8px 14px; }
  .composer textarea { font-size: 16px; }
  .doc-chip { display: none; }
  .icon, .send { width: 40px; height: 40px; }
  .lang { max-width: 64px; }
  .composer-meta { font-size: 11.5px; text-align: center; }
  .sidebar, .evidence { width: 100vw; }
  .excerpt p { font-size: 14.5px; }
  .toast { bottom: 90px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
</style>
</head>
<body>
<div id="root"></div>

<script src="https://cdn.jsdelivr.net/npm/react@18.3.1/umd/react.production.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/react-dom@18.3.1/umd/react-dom.production.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@babel/standalone@7.26.0/babel.min.js"></script>

<script type="text/babel">
const { useState, useEffect, useRef, useCallback } = React;
const MAX_Q = 2000;

/* ---------- API ---------- */
async function parse(res) {
  if (!res.ok) {
    let message = 'Request failed (' + res.status + ')';
    try { const d = await res.json(); if (typeof d.detail === 'string') message = d.detail; } catch (e) {}
    throw new Error(message);
  }
  return res.json();
}
const api = {
  status: () => fetch('/api/status').then(parse),
  ask: (question, k, signal) => fetch('/api/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, k }), signal }).then(parse),
  upload: (file, replace) => { const f = new FormData(); f.append('file', file); f.append('replace', replace ? 'true' : 'false'); return fetch('/api/upload', { method: 'POST', body: f }).then(parse); },
  clear: () => fetch('/api/clear', { method: 'POST' }).then(parse),
};
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

/* ---------- Answer: paragraphs, lists, headings, **bold**, `code`, math (KaTeX), [Source N] -> button ---------- */
function renderMath(tex, display, key) {
  try {
    const html = window.katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: false });
    return <span key={key} className={display ? 'math-block' : 'math-inline'} dangerouslySetInnerHTML={{ __html: html }} />;
  } catch (e) { return <code key={key}>{tex}</code>; }
}
const TOKEN = /(\\\((?:.|\n)+?\\\)|\$\$[\s\S]+?\$\$|\$(?=\S)[^$\n]*?\S\$(?!\d)|\*\*.+?\*\*|`[^`]+`)/g;
function inline(text, onCite, keyBase) {
  const out = [];
  const cite = /[\[【]([^\]】]*Source[^\]】]*)[\]】]/g;
  let last = 0, m, i = 0;
  const pushText = (str) => {
    str.split(TOKEN).forEach((part) => {
      if (!part) return;
      const k = keyBase + '-t' + i++;
      if (part.startsWith('\\(') && part.endsWith('\\)')) out.push(renderMath(part.slice(2, -2).trim(), false, k));
      else if (part.startsWith('$$') && part.endsWith('$$') && part.length > 4) out.push(renderMath(part.slice(2, -2).trim(), false, k));
      else if (part.startsWith('$') && part.endsWith('$') && part.length > 2) out.push(renderMath(part.slice(1, -1).trim(), false, k));
      else if (part.startsWith('**') && part.endsWith('**') && part.length > 4) out.push(<strong key={k}>{part.slice(2, -2)}</strong>);
      else if (part.startsWith('`') && part.endsWith('`') && part.length > 2) out.push(<code key={k}>{part.slice(1, -1)}</code>);
      else out.push(part);
    });
  };
  while ((m = cite.exec(text))) {
    if (m.index > last) pushText(text.slice(last, m.index));
    (m[1].match(/\d+/g) || []).forEach((n) => {
      out.push(<button key={keyBase + '-c' + i++} className="cite" onClick={() => onCite(Number(n))} aria-label={'Show source ' + n}>{n}</button>);
    });
    last = cite.lastIndex;
  }
  if (last < text.length) pushText(text.slice(last));
  return out;
}
function renderBlock(block, key, onCite) {
  const lines = block.split('\n');
  const h = lines.length === 1 && /^#{1,4}\s+(.*)$/.exec(lines[0]);
  if (h) return <h3 key={key}>{inline(h[1], onCite, key)}</h3>;
  if (lines.every((l) => /^\s*[-*•]\s+/.test(l))) {
    return <ul key={key}>{lines.map((l, li) => <li key={li}>{inline(l.replace(/^\s*[-*•]\s+/, ''), onCite, key + '-' + li)}</li>)}</ul>;
  }
  if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) {
    return <ol key={key}>{lines.map((l, li) => <li key={li}>{inline(l.replace(/^\s*\d+[.)]\s+/, ''), onCite, key + '-' + li)}</li>)}</ol>;
  }
  return <p key={key}>{lines.map((l, li) => <span key={li}>{li > 0 && <br />}{inline(l, onCite, key + '-' + li)}</span>)}</p>;
}
function Answer({ text, onCite }) {
  const parts = text.trim().split(/(\\\[[\s\S]+?\\\]|\$\$[\s\S]+?\$\$)/g);
  return (
    <div className="answer">
      {parts.map((part, pi) => {
        if (pi % 2) return <div key={pi}>{renderMath(part.slice(2, -2).trim(), true, 'm' + pi)}</div>;
        if (!part.trim()) return null;
        return part.trim().split(/\n{2,}/).map((block, bi) => renderBlock(block, pi + '-' + bi, onCite));
      })}
    </div>
  );
}

/* ---------- Sidebar ---------- */
function Sidebar({ status, k, setK, onUpload, onClear, onNew, onExport, canExport, theme, onTheme, open, onClose }) {
  const [file, setFile] = useState(null);
  const [replace, setReplace] = useState(true);
  const [working, setWorking] = useState(false);
  const [over, setOver] = useState(false);
  const [notice, setNotice] = useState(null);
  const inputRef = useRef(null);

  function pick(f) {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith('.pdf')) { setNotice({ ok: false, text: 'Only PDF files are supported.' }); return; }
    setNotice(null); setFile(f);
  }
  function onDrop(e) {
    e.preventDefault(); setOver(false);
    pick(e.dataTransfer.files && e.dataTransfer.files[0]);
  }
  async function handleUpload() {
    if (!file) return;
    setWorking(true); setNotice(null);
    try {
      const r = await onUpload(file, replace);
      setNotice({ ok: true, text: 'Indexed ' + r.pages + ' pages as ' + r.chunks + ' passages. New passages can take up to 20 seconds to become searchable.' });
      setFile(null);
      if (inputRef.current) inputRef.current.value = '';
    } catch (e) { setNotice({ ok: false, text: e.message }); }
    finally { setWorking(false); }
  }

  return (
    <aside className={'sidebar' + (open ? ' open' : '')} aria-label="Document and settings">
      <div className="brand">
        <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
          <rect x="3" y="2" width="20" height="22" rx="3" fill="none" stroke="currentColor" strokeWidth="1.8" />
          <rect x="7" y="8" width="12" height="4" rx="1" className="mark-fill" />
          <path d="M7 16h12M7 19.5h8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
        <span>Glean</span>
        <button className="close" onClick={onClose} aria-label="Close panel">×</button>
      </div>

      <div className="row">
        <button className="btn ghost" onClick={onNew}>New chat</button>
        <button className="btn ghost" onClick={onExport} disabled={!canExport} title="Download this conversation as Markdown">Export</button>
      </div>

      <section>
        <h2 className="side-title">Document</h2>
        <label className={'drop' + (file ? ' has-file' : '') + (over ? ' over' : '')}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)} onDrop={onDrop}>
          <input ref={inputRef} type="file" accept="application/pdf" onChange={(e) => pick(e.target.files && e.target.files[0])} />
          {file ? (<><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(1)} MB</span></>)
                : (<><strong>Choose a PDF</strong><span>or drop one here</span></>)}
        </label>
        <label className="check">
          <input type="checkbox" checked={replace} onChange={(e) => setReplace(e.target.checked)} />
          Replace what is already indexed
        </label>
        <button className="btn" onClick={handleUpload} disabled={!file || working}>{working ? 'Indexing…' : 'Index PDF'}</button>
        {notice && <p className={'notice ' + (notice.ok ? 'ok' : 'bad')} role="status">{notice.text}</p>}
      </section>

      <section>
        <h2 className="side-title">Retrieval</h2>
        <label className="range">
          <span>Passages per answer <b>{k}</b></span>
          <input type="range" min="1" max="8" value={k} onChange={(e) => setK(Number(e.target.value))} />
        </label>
        <p className="hint">More passages give broader answers; fewer give tighter ones.</p>
      </section>

      <section className="push-down">
        <h2 className="side-title">Index</h2>
        <dl className="stats">
          <div><dt>Name</dt><dd>{status.index || '–'}</dd></div>
          <div><dt>Passages stored</dt><dd>{status.vectors == null ? '–' : status.vectors}</dd></div>
          <div><dt>Server</dt><dd className={status.online ? 'on' : 'off'}><i /> {status.online ? 'Connected' : 'Offline'}</dd></div>
        </dl>
        <div className="row">
          <button className="btn ghost" onClick={onTheme}>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</button>
          <button className="btn ghost danger" onClick={onClear} disabled={!status.online}>Clear index</button>
        </div>
      </section>
    </aside>
  );
}

/* ---------- Thread + composer ---------- */
const SUGGESTIONS = [
  'What problem does this paper set out to solve?',
  'Summarize the main method in a few sentences.',
  'Which results does the paper report?',
];
function Icon({ children, size = 18 }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{children}</svg>;
}
function Thread({ messages, activeId, busy, online, vectors, onSend, onCite, onRetry, onCopy, onStop, onNotify, onQuickUpload, docName, uploading, voice, onVoice, onSpeak, speakingId }) {
  const [text, setText] = useState('');
  const endRef = useRef(null);
  const areaRef = useRef(null);
  const [menu, setMenu] = useState(false);
  const [lang, setLang] = useState('en-IN');
  const [listening, setListening] = useState(false);
  const recRef = useRef(null);
  const fileRef = useRef(null);
  const replaceRef = useRef(false);
  const textRef = useRef('');
  const sendRef = useRef(onSend); sendRef.current = onSend;
  const voiceRef = useRef(voice); voiceRef.current = voice;

  useEffect(() => {
    if (!menu) return;
    const close = () => setMenu(false);
    document.addEventListener('click', close);
    return () => document.removeEventListener('click', close);
  }, [menu]);

  function pickFile(replace) { replaceRef.current = replace; setMenu(false); fileRef.current && fileRef.current.click(); }
  function toggleMic() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { onNotify('Voice input works in Chrome or Edge'); return; }
    if (listening && recRef.current) { recRef.current.stop(); return; }
    const base = text.trim() ? text.trim() + ' ' : '';
    const r = new SR();
    r.lang = lang; r.interimResults = true; r.continuous = false;
    textRef.current = base;
    r.onresult = (e) => {
      let t = '';
      for (let i = 0; i < e.results.length; i++) t += e.results[i][0].transcript;
      textRef.current = base + t; setText(base + t);
    };
    r.onerror = (e) => { setListening(false); onNotify(e.error === 'not-allowed' ? 'Allow microphone access to use voice input' : 'Voice input stopped'); };
    r.onend = () => {
      setListening(false);
      const q = textRef.current.trim();
      if (voiceRef.current && q) { setText(''); textRef.current = ''; sendRef.current(q); }
    };
    recRef.current = r; setListening(true); r.start();
  }

  useEffect(() => { endRef.current && endRef.current.scrollIntoView({ block: 'end' }); }, [messages]);
  useEffect(() => {
    const el = areaRef.current; if (!el) return;
    el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 140) + 'px';
  }, [text]);

  const tooLong = text.length > MAX_Q;
  function submit() {
    const q = text.trim(); if (!q || busy || tooLong) return;
    setText(''); onSend(q);
  }

  return (
    <>
      <div className="thread"><div className="thread-inner">
        {messages.length === 0 && (
          <div className="empty">
            <h1>Upload a PDF, then ask it a question.</h1>
            <p>Every answer links back to the passages it came from.</p>
            {!online && <p className="notice bad">The server is offline. Start it again with <code>py main.py</code>.</p>}
            {online && vectors === 0 && <p className="notice bad">Nothing is indexed yet. Add a PDF from the Document panel.</p>}
            <ul className="suggest">
              {SUGGESTIONS.map((s) => <li key={s}><button onClick={() => onSend(s)} disabled={busy}>{s}</button></li>)}
            </ul>
          </div>
        )}
        {messages.map((m) => (
          <article key={m.id} className={'turn' + (m.id === activeId ? ' is-active' : '')}>
            <h2 className="question">{m.question}</h2>
            {m.error ? <p className="notice bad">{m.error}<button className="link" onClick={() => onRetry(m.id)} disabled={busy}>Try again</button></p>
              : m.answer === null ? <div className="typing" role="status" aria-label="Reading the document"><i /><i /><i /></div>
              : (<>
                  <Answer text={m.answer} onCite={(n) => onCite(m.id, n)} />
                  <div className="actions">
                    {m.sources.length > 0 && <button className="link main" onClick={() => onCite(m.id, null)}>{m.sources.length} passages used</button>}
                    <button className="link" onClick={() => onCopy(m.answer)}>Copy</button>
                    <button className="link" onClick={() => onSpeak(m.answer, m.id)}>{speakingId === m.id ? 'Stop listening' : 'Listen'}</button>
                    <button className="link" onClick={() => onRetry(m.id)} disabled={busy}>Regenerate</button>
                  </div>
                </>)}
          </article>
        ))}
        <div ref={endRef} />
      </div></div>

      <div className="composer-wrap">
        <div className="composer">
          <textarea ref={areaRef} rows={1} value={text} placeholder={listening ? 'Listening…' : 'Ask anything about your document'} aria-label="Your question"
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && !window.matchMedia('(pointer: coarse)').matches) { e.preventDefault(); submit(); } }} />
          <div className="tools">
            <div className="menu-wrap">
              <button className="icon" onClick={(e) => { e.stopPropagation(); setMenu(!menu); }} aria-label="Add a PDF" aria-expanded={menu}><Icon><path d="M12 5v14M5 12h14" /></Icon></button>
              {menu && <div className="menu" role="menu">
                <button role="menuitem" onClick={() => pickFile(false)}>Add PDF to index</button>
                <button role="menuitem" onClick={() => pickFile(true)}>Replace index with PDF</button>
              </div>}
            </div>
            <input ref={fileRef} type="file" accept="application/pdf" hidden onChange={(e) => { const f = e.target.files && e.target.files[0]; e.target.value = ''; if (f) onQuickUpload(f, replaceRef.current); }} />
            <span className="doc-chip" title={docName || ''}>{uploading ? 'Indexing…' : (docName || 'Indexed document')}</span>
            <span className="grow" />
            <button className={'icon' + (listening ? ' on' : '')} onClick={toggleMic} aria-label="Voice input" aria-pressed={listening}><Icon><rect x="9" y="3" width="6" height="11" rx="3" /><path d="M5 11a7 7 0 0 0 14 0M12 18v3" /></Icon></button>
            <select className="lang" value={lang} onChange={(e) => setLang(e.target.value)} aria-label="Voice language">
              <option value="en-IN">English (IN)</option><option value="hi-IN">हिन्दी</option><option value="en-US">English (US)</option>
            </select>
            <button className={'icon' + (voice ? ' on' : '')} onClick={onVoice} aria-label="Voice mode: read answers aloud" aria-pressed={voice} title="Voice mode: answers are read aloud"><Icon><path d="M4 10v4M8 6v12M12 3v18M16 8v8M20 11v2" /></Icon></button>
            {busy
              ? <button className="send" onClick={onStop} aria-label="Stop"><Icon size={16}><rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" /></Icon></button>
              : <button className="send" onClick={submit} disabled={!text.trim() || tooLong} aria-label="Send"><Icon><path d="M12 19V5M5 12l7-7 7 7" /></Icon></button>}
          </div>
        </div>
        <div className="composer-meta">
          <span>AI can make mistakes. Verify important answers against the original PDF.</span>
          {text.length > 1500 && <span className={tooLong ? 'over' : ''}>{text.length} / {MAX_Q}</span>}
        </div>
      </div>
    </>
  );
}

/* ---------- Evidence ---------- */
function Evidence({ message, activeN, onSelect, open, onClose }) {
  const refs = useRef({});
  useEffect(() => {
    if (activeN == null) return;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const el = refs.current[activeN];
    if (el) el.scrollIntoView({ block: 'nearest', behavior: reduce ? 'auto' : 'smooth' });
  }, [activeN, message && message.id]);

  const sources = message ? message.sources : [];
  return (
    <aside className={'evidence' + (open ? ' open' : '')} aria-label="Evidence">
      <div className="evidence-head"><h2>Evidence</h2><button className="close" onClick={onClose} aria-label="Close panel">×</button></div>
      {sources.length === 0 ? (
        <p className="muted">Passages behind an answer appear here. Select a numbered citation to jump to its source.</p>
      ) : (
        <>
          <p className="muted ask-ref">For: {message.question}</p>
          {sources.map((s) => (
            <section key={s.n} ref={(el) => (refs.current[s.n] = el)} className={'excerpt' + (activeN === s.n ? ' is-on' : '')} onClick={() => onSelect(s.n)}>
              <header><span className="badge">{s.n}</span><span>{s.page ? 'Page ' + s.page : 'Page unknown'}</span></header>
              <p>{s.text.replace(/\s+/g, ' ')}</p>
            </section>
          ))}
        </>
      )}
    </aside>
  );
}

/* ---------- App ---------- */
function App() {
  const [messages, setMessages] = useState([]);
  const [active, setActive] = useState(null);
  const [status, setStatus] = useState({ online: true, vectors: null, index: '' });
  const [k, setK] = useState(4);
  const [busy, setBusy] = useState(false);
  const [panel, setPanel] = useState(null);
  const [toast, setToast] = useState(null);
  const [theme, setTheme] = useState(document.documentElement.getAttribute('data-theme') || 'light');
  const toastTimer = useRef(null);
  const [showSide, setShowSide] = useState(() => { try { return localStorage.getItem('marginalia-sidebar') !== 'off'; } catch (e) { return true; } });
  function toggleSide() {
    const next = !showSide;
    setShowSide(next);
    try { localStorage.setItem('marginalia-sidebar', next ? 'on' : 'off'); } catch (e) {}
  }
  const [docName, setDocName] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [voice, setVoice] = useState(false);
  const [speakingId, setSpeakingId] = useState(null);
  const voiceRef = useRef(false);
  const abortRef = useRef(null);
  useEffect(() => { voiceRef.current = voice; }, [voice]);
  const plain = (t) => t.replace(/[\[【][^\]】]*Source[^\]】]*[\]】]/g, '').replace(/\\[\[\]()]|\$/g, '').replace(/[*`#]/g, '').replace(/\s+/g, ' ').trim();
  function speak(text, id) {
    if (!('speechSynthesis' in window)) { notify('Read aloud is not supported in this browser'); return; }
    const synth = window.speechSynthesis;
    if (id != null && speakingId === id) { synth.cancel(); setSpeakingId(null); return; }
    synth.cancel();
    const u = new SpeechSynthesisUtterance(plain(text));
    u.lang = /[\u0900-\u097F]/.test(text) ? 'hi-IN' : 'en-IN';
    u.onend = u.onerror = () => setSpeakingId(null);
    setSpeakingId(id); synth.speak(u);
  }
  function toggleVoice() {
    if (voice && 'speechSynthesis' in window) { window.speechSynthesis.cancel(); setSpeakingId(null); }
    setVoice(!voice);
    notify(voice ? 'Voice mode off' : 'Voice mode on: answers are read aloud, and the mic sends when you stop speaking');
  }
  function stop() { abortRef.current && abortRef.current.abort(); }
  const [showEv, setShowEv] = useState(() => { try { return localStorage.getItem('marginalia-evidence') !== 'off'; } catch (e) { return true; } });
  function toggleEv() {
    const next = !showEv;
    setShowEv(next);
    try { localStorage.setItem('marginalia-evidence', next ? 'on' : 'off'); } catch (e) {}
  }

  const notify = useCallback((text) => {
    setToast(text);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 2200);
  }, []);

  const refreshStatus = useCallback(async () => {
    try { const s = await api.status(); setStatus({ online: true, vectors: s.vectors, index: s.index }); }
    catch (e) { setStatus((p) => ({ ...p, online: false })); }
  }, []);
  useEffect(() => {
    refreshStatus();
    const t = setInterval(refreshStatus, 30000);
    return () => clearInterval(t);
  }, [refreshStatus]);

  function toggleTheme() {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('marginalia-theme', next); } catch (e) {}
  }

  async function run(id, question) {
    setBusy(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      const r = await api.ask(question, k, ctrl.signal);
      setMessages((m) => m.map((x) => (x.id === id ? { ...x, answer: r.answer, sources: r.sources, error: null } : x)));
      if (voiceRef.current) speak(r.answer, id);
    } catch (e) {
      const msg = e.name === 'AbortError' ? 'Stopped. You can try again.' : e.message;
      setMessages((m) => m.map((x) => (x.id === id ? { ...x, error: msg } : x)));
    } finally { setBusy(false); }
  }
  function send(question) {
    const id = uid();
    setMessages((m) => [...m, { id, question, answer: null, sources: [], error: null }]);
    setActive({ id, n: null });
    run(id, question);
  }
  function retry(id) {
    const msg = messages.find((x) => x.id === id);
    if (!msg || busy) return;
    setMessages((m) => m.map((x) => (x.id === id ? { ...x, answer: null, sources: [], error: null } : x)));
    setActive({ id, n: null });
    run(id, msg.question);
  }
  async function handleUpload(file, replace) {
    const r = await api.upload(file, replace);
    setDocName(file.name);
    refreshStatus(); setTimeout(refreshStatus, 8000);
    notify('Indexed ' + r.filename);
    return r;
  }
  async function handleClear() {
    if (!window.confirm('Remove every indexed passage from this index?')) return;
    try { await api.clear(); setDocName(null); notify('Index cleared'); }
    catch (e) { notify(e.message); }
    refreshStatus();
  }
  async function quickUpload(file, replace) {
    if (!file.name.toLowerCase().endsWith('.pdf')) { notify('Only PDF files are supported'); return; }
    setUploading(true);
    try { await handleUpload(file, replace); }
    catch (e) { notify(e.message); }
    finally { setUploading(false); }
  }
  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); notify('Answer copied'); }
    catch (e) { notify('Copy is blocked by the browser'); }
  }
  function exportChat() {
    const done = messages.filter((m) => m.answer);
    if (!done.length) return;
    const body = done.map((m) => {
      const refs = m.sources.map((s) => '- [' + s.n + '] ' + (s.page ? 'Page ' + s.page : 'Page unknown') + ': ' + s.text.replace(/\s+/g, ' ').slice(0, 300)).join('\n');
      return '## ' + m.question + '\n\n' + m.answer.trim() + (refs ? '\n\n**Sources**\n' + refs : '');
    }).join('\n\n---\n\n');
    const blob = new Blob(['# Glean conversation\n\n' + body + '\n'], { type: 'text/markdown' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'marginalia-conversation.md';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    notify('Conversation exported');
  }
  function showSources(id, n = null) { setActive({ id, n }); setPanel('evidence'); setShowEv(true); }

  const activeMessage = messages.find((m) => m.id === (active && active.id)) || [...messages].reverse().find((m) => m.sources.length);

  return (
    <div className={'app' + (showEv ? '' : ' no-evidence') + (showSide ? '' : ' no-side')}>
      <Sidebar status={status} k={k} setK={setK} onUpload={handleUpload} onClear={handleClear}
        onNew={() => { setMessages([]); setActive(null); setPanel(null); }}
        onExport={exportChat} canExport={messages.some((m) => m.answer)}
        theme={theme} onTheme={toggleTheme}
        open={panel === 'doc'} onClose={() => setPanel(null)} />
      <div className="conversation">
        <div className="convbar">
          <button className="icon" onClick={toggleSide} aria-label={showSide ? 'Hide sidebar' : 'Show sidebar'} aria-pressed={showSide}><Icon><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M9 4v16" /></Icon></button>
          <span className="conv-title">Glean</span>
          <span className="grow" />
          <span className={'pill' + (status.online ? '' : ' off')}><i />{status.online ? (status.vectors != null ? status.vectors + ' passages indexed' : 'Connected') : 'Offline'}</span>
          <button className="icon" onClick={toggleTheme} aria-label="Switch light or dark theme"><Icon><circle cx="12" cy="12" r="9" /><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor" /></Icon></button>
          <button className="btn ghost sm" onClick={toggleEv} aria-pressed={showEv}>{showEv ? 'Hide evidence' : 'Show evidence'}</button>
        </div>
        <div className="topbar">
          <span className="brand-sm">Glean</span>
          <div className="topbar-actions">
            <button className="icon" onClick={toggleTheme} aria-label="Switch light or dark theme"><Icon><circle cx="12" cy="12" r="9" /><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor" /></Icon></button>
            <button className="btn ghost sm" onClick={() => setPanel('doc')}>Document</button>
            <button className="btn ghost sm" onClick={() => setPanel('evidence')}>Evidence</button>
          </div>
        </div>
        <Thread messages={messages} activeId={activeMessage && activeMessage.id} busy={busy}
          online={status.online} vectors={status.vectors} onSend={send} onCite={showSources}
          onRetry={retry} onCopy={copyText} onStop={stop} onNotify={notify}
          onQuickUpload={quickUpload} docName={docName} uploading={uploading}
          voice={voice} onVoice={toggleVoice} onSpeak={speak} speakingId={speakingId} />
      </div>
      <Evidence message={activeMessage}
        activeN={active && activeMessage && active.id === activeMessage.id ? active.n : null}
        onSelect={(n) => activeMessage && setActive({ id: activeMessage.id, n })}
        open={panel === 'evidence'} onClose={() => setPanel(null)} />
      {panel && <div className="scrim" onClick={() => setPanel(null)} />}
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
</script>
</body>
</html>
'''


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    URL = "http://127.0.0.1:8000"
    threading.Timer(1.5, lambda: webbrowser.open(URL)).start()
    print(f"Glean chal raha hai: {URL}  (band karne ke liye Ctrl+C)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
