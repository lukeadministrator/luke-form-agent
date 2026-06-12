# luke-form-agent

A chat-driven form builder. Users describe a form in natural language; an LLM
returns the **complete form definition as JSON each turn**, validated with
Pydantic, and rendered live.

- **Backend:** FastAPI (Python)
- **Brain:** Groq free tier in production; Gemini or local Ollama optional
- **Output contract:** full `Form` JSON per turn (see `app/schema.py`)
- **Cost:** $0 — Render free plan + Groq free tier

> Later, the `Form` JSON maps onto the luke-capability-engine form-definition
> format (`formKey` / versions). For now it is standalone.

## Architecture

```
browser chat ──POST /chat──▶ FastAPI ──▶ llm.generate_form()
     ▲                                        │  Groq (prod) or Ollama (dev)
     └────────── updated Form JSON ◀──────────┘  validated by Pydantic
```

The agent is **stateless**: the client sends the current form back with each
message, so no database is needed yet.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # paste a Groq key, OR leave blank + run Ollama
uvicorn app.main:app --reload
# open http://localhost:8000
```

Get a free Groq key (email signup, no card): https://console.groq.com/keys

### No key? Use a local model instead

```bash
# install Ollama from https://ollama.com
ollama pull qwen2.5:7b
# uncomment `ollama` in requirements.txt and `pip install ollama`
# leave GROQ_API_KEY blank — the app auto-falls back to Ollama
```

## Deploy to Render

1. Push this folder to a GitHub repo.
2. In Render: **New ➜ Blueprint**, point it at the repo (`render.yaml` is picked up).
3. Add the `GROQ_API_KEY` env var in the dashboard.
4. Deploy. Visit the service URL — the test client loads at `/`.

> Note: the free plan can't run local models (too little RAM), which is why
> production uses the hosted Groq free tier.

## API

`POST /chat`

```json
{ "message": "add a required email field", "form": { "title": "Signup", "fields": [] } }
```

→

```json
{ "form": { "title": "Signup", "fields": [ { "key": "email", "label": "Email", "type": "email", "required": true, "options": null, "placeholder": null } ] }, "brain": "groq" }
```

`GET /health` → `{ "status": "ok", "brain": "groq" }`
