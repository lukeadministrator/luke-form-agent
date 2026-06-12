"""FastAPI entrypoint for the form-building agent.

Endpoints:
  GET  /          -> tiny browser test client
  GET  /health    -> liveness + which brain is active
  POST /chat      -> {message, schema?, title?} -> {schema, title, brain}

`schema` in/out is the coltorapps builder schema ({entities, root}) that
luke-consumer-ui's FormRenderer / builderStore and luke-capability-engine's
draftSchema consume directly.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .coltorapps import schema_to_spec, spec_to_schema
from .llm import active_brain, generate_spec
from .schema import ChatRequest, ChatResponse

load_dotenv()

app = FastAPI(title="luke-form-agent", version="0.2.0")

# Allow the browser-based Form Builder (consumer-ui) to call us. Set
# FORM_AGENT_CORS to a comma-separated list of origins in prod to lock it down.
_origins = os.getenv("FORM_AGENT_CORS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins.strip() == "*" else [o.strip() for o in _origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = Path(__file__).parent / "static" / "index.html"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "brain": active_brain()}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # Project the incoming coltorapps schema to a flat spec, keeping the bits we
    # must not lose so the rebuild can merge instead of clobber.
    spec, existing, preserved_entities, preserved_root_ids = schema_to_spec(req.schema)
    if req.title:
        spec.title = req.title

    try:
        updated = generate_spec(spec, req.message)
    except Exception as exc:  # invalid JSON, model/network error, etc.
        raise HTTPException(status_code=502, detail=f"brain error: {exc}") from exc

    out_schema = spec_to_schema(updated, existing, preserved_entities, preserved_root_ids)
    return ChatResponse(schema=out_schema, title=updated.title, brain=active_brain())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    if STATIC.exists():
        return STATIC.read_text(encoding="utf-8")
    return "<h1>luke-form-agent</h1><p>POST /chat to build forms.</p>"
