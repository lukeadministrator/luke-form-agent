"""FastAPI entrypoint for the form-building agent.

Endpoints:
  GET  /          -> tiny browser test client (so you can try it immediately)
  GET  /health    -> liveness + which brain is active
  POST /chat      -> {message, form?} -> {form, brain}
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .llm import active_brain, generate_form
from .schema import ChatRequest, ChatResponse, Form

load_dotenv()

app = FastAPI(title="luke-form-agent", version="0.1.0")

STATIC = Path(__file__).parent / "static" / "index.html"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "brain": active_brain()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    current = req.form or Form()
    try:
        form = generate_form(current, req.message)
    except Exception as exc:  # invalid JSON, model/network error, etc.
        raise HTTPException(status_code=502, detail=f"brain error: {exc}") from exc
    return ChatResponse(form=form, brain=active_brain())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    if STATIC.exists():
        return STATIC.read_text(encoding="utf-8")
    return "<h1>luke-form-agent</h1><p>POST /chat to build forms.</p>"
