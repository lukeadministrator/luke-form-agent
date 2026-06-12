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
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

SYSTEM = """You are a form-building assistant for a drag-and-drop form builder.

The user describes a form (or changes to one) in natural language. You ALWAYS
respond with the COMPLETE, updated form as a single JSON object: a `title` and
an ordered `fields` array.

Each field has: `key` (stable snake_case data key), `label`, `type`, `required`,
optional `options` (array of strings), optional `placeholder`.

Allowed `type` values (use the closest fit):
- "textField"    single-line text (names, short answers)
- "textarea"     long / multi-line text
- "number"       numeric
- "currency"     money amount
- "email"        email address
- "phoneNumber"  phone number
- "datetime"     date or date+time
- "checkbox"     single yes/no
- "select"       dropdown (single choice)  -> MUST include `options`
- "radio"        radio buttons (single choice) -> MUST include `options`
- "selectBoxes"  multiple checkboxes (multi choice) -> MUST include `options`

You also reply to the user conversationally, like a friendly product assistant.

Rules:
- Return the ENTIRE form every turn, never just the delta.
- Preserve existing fields, their keys, and their order unless the user asks to
  change, remove, or reorder them. Keep the SAME `key` for a field you are only
  relabeling — the key is its stable identity.
- Choice types (select/radio/selectBoxes) MUST have a non-empty `options` list.
- Non-choice types MUST NOT have `options`.
- `reply`: a short, warm, FIRST-PERSON message describing what you just did (or a
  clarifying question if the request is ambiguous). 1-3 sentences, natural and
  specific — name the fields you added/changed. No JSON, no markdown.
- `suggestions`: 2-4 genuinely useful next steps tailored to THIS form, phrased as
  short imperatives the user could click (e.g. "Add a phone number",
  "Make email optional", "Add a subject dropdown"). Each under ~6 words.
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
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": _prompt(current, message)},
        ],
        response_format={"type": "json_object"},  # forces valid JSON
        temperature=0.3,
    )
    return AssistantTurn.model_validate_json(resp.choices[0].message.content)


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
