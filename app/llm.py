"""LLM brain.

Three interchangeable backends, picked at runtime (first match wins):
  * Groq free tier   -> used on Render / anywhere with GROQ_API_KEY (default).
  * Gemini free tier -> used if GEMINI_API_KEY is set (blocked for managed domains).
  * Ollama           -> local open model in dev when no cloud key is set.

All are driven the same way: we hand the model the *current* form (as a flat
FormSpec) plus the user's request and demand the *complete* updated FormSpec
back as JSON. We validate with Pydantic before returning. Python then renders
the FormSpec into the coltorapps builder schema (see coltorapps.py).
"""
from __future__ import annotations

import json
import os

from .schema import AssistantTurn, FormSpec

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "moonshotai/kimi-k2-instruct")
# Falls back to this if the primary model errors (bad id, rate limit, bad JSON).
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

SYSTEM = """You are LukeTalks, a friendly assistant that builds and edits forms in
a drag-and-drop form builder.

You are given the CURRENT form and a user message. You ALWAYS respond with a
single JSON object: the COMPLETE form (`title` + ordered `fields`), a
conversational `reply`, and a few `suggestions`.

FIELD MODEL
Each field: `key` (stable snake_case id), `label`, `type`, `required`,
optional `options` (string array — choice types only), optional `placeholder`.
Allowed `type` (pick the closest fit):
- textField (short text), textarea (long text), number, currency, email,
  phoneNumber, datetime
- checkbox (single yes/no)
- select (dropdown — needs options), radio (needs options),
  selectBoxes (multi-select — needs options)
- button (a clickable button such as a Submit button; give it a label like
  "Submit". A button has no required/options/placeholder.)

EDIT vs. CHAT — decide first which the message is:
- An EDIT ("add a phone number", "make email optional", "remove subject",
  "change to radio buttons"): update `fields` accordingly.
- A QUESTION or off-topic / chit-chat ("what can you do?", "can you write
  JavaScript?", "thanks"): DO NOT change the form — return `fields` EXACTLY as
  given — and answer in `reply`. You build forms; you can't run code or do tasks
  unrelated to forms, so say so briefly and steer back to the form. NEVER invent
  a field to satisfy a non-form request (e.g. do not add a "JavaScript" field).

RULES
- Always return the ENTIRE form, never a partial.
- Preserve existing fields, keys, and order unless asked to change them. Keep the
  SAME `key` when only relabeling — the key is the field's identity.
- Choice types MUST have a non-empty `options`; other types MUST NOT have options.
- `reply`: 1-2 sentences, warm and SPECIFIC about what changed (name the fields).
  VARY your wording every turn. Do NOT end with boilerplate like "let me know if
  you need any further changes" — just say what you did, naturally.
- `suggestions`: 2-4 genuinely useful next steps for THIS form, as short
  imperatives (under ~6 words). For a question, suggest relevant form actions.
- Output ONLY the JSON object — no prose, no markdown fences."""


def _prompt(current: FormSpec, message: str) -> str:
    return (
        f"Current form (JSON):\n{current.model_dump_json(indent=2)}\n\n"
        f"User request:\n{message}"
    )


def active_brain() -> str:
    if GROQ_API_KEY:
        return "groq"
    if GEMINI_API_KEY:
        return "gemini"
    return "ollama"


def generate_turn(current: FormSpec, message: str) -> AssistantTurn:
    brain = active_brain()
    if brain == "groq":
        return _groq(current, message)
    if brain == "gemini":
        return _gemini(current, message)
    return _ollama(current, message)


def _groq(current: FormSpec, message: str) -> AssistantTurn:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    # Groq JSON mode needs the target shape in the prompt, so embed the schema.
    system = (
        f"{SYSTEM}\n\nThe JSON object MUST conform to this JSON Schema:\n"
        f"{json.dumps(AssistantTurn.model_json_schema())}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _prompt(current, message)},
    ]
    # Try the primary model, then fall back to a known-good one on any error
    # (unavailable model id, rate limit, malformed JSON, …).
    models = [GROQ_MODEL]
    if GROQ_FALLBACK_MODEL and GROQ_FALLBACK_MODEL != GROQ_MODEL:
        models.append(GROQ_FALLBACK_MODEL)

    last_err: Exception | None = None
    for model in models:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            return AssistantTurn.model_validate_json(resp.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001 - try the next model
            last_err = exc
    raise last_err  # type: ignore[misc]


def _gemini(current: FormSpec, message: str) -> AssistantTurn:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_prompt(current, message),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=AssistantTurn,  # forces schema-shaped JSON
            temperature=0.3,
        ),
    )
    return AssistantTurn.model_validate_json(resp.text)


def _ollama(current: FormSpec, message: str) -> AssistantTurn:
    import ollama

    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _prompt(current, message)},
        ],
        format=AssistantTurn.model_json_schema(),  # forces schema-shaped JSON
        options={"temperature": 0.3},
    )
    return AssistantTurn.model_validate_json(resp["message"]["content"])
