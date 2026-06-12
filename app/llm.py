"""LLM brain.

Three interchangeable backends, picked at runtime (first match wins):
  * Groq free tier   -> used on Render / anywhere with GROQ_API_KEY (default).
  * Gemini free tier -> used if GEMINI_API_KEY is set (blocked for managed domains).
  * Ollama           -> local open model in dev when no cloud key is set.

All are driven the same way: we hand the model the *current* form plus the
user's request and demand the *complete* updated form back as JSON matching
the Form schema. We then validate with Pydantic before returning.
"""
from __future__ import annotations

import json
import os

from .schema import Form

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

SYSTEM = """You are a form-building assistant.

The user describes a form (or changes to one) in natural language. You ALWAYS
respond with the COMPLETE, updated form as a single JSON object matching the
required schema.

Rules:
- Return the ENTIRE form every turn, never just the delta.
- Preserve existing fields and their order unless the user asks to change,
  remove, or reorder them.
- Give every field a stable snake_case `key`.
- `dropdown` fields MUST include a non-empty `options` list.
- Infer sensible field `type`s (an email field -> "email", a long answer ->
  "textarea", a yes/no -> "checkbox", a date -> "date").
- Output ONLY the JSON object — no prose, no markdown fences."""


def _prompt(current: Form, message: str) -> str:
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


def generate_form(current: Form, message: str) -> Form:
    brain = active_brain()
    if brain == "groq":
        return _groq(current, message)
    if brain == "gemini":
        return _gemini(current, message)
    return _ollama(current, message)


def _groq(current: Form, message: str) -> Form:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    # Groq JSON mode needs the target shape in the prompt, so embed the schema.
    system = (
        f"{SYSTEM}\n\nThe JSON object MUST conform to this JSON Schema:\n"
        f"{json.dumps(Form.model_json_schema())}"
    )
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": _prompt(current, message)},
        ],
        response_format={"type": "json_object"},  # forces valid JSON
        temperature=0.2,
    )
    return Form.model_validate_json(resp.choices[0].message.content)


def _gemini(current: Form, message: str) -> Form:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=_prompt(current, message),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=Form,  # forces schema-shaped JSON
            temperature=0.2,
        ),
    )
    return Form.model_validate_json(resp.text)


def _ollama(current: Form, message: str) -> Form:
    import ollama

    resp = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _prompt(current, message)},
        ],
        format=Form.model_json_schema(),  # forces schema-shaped JSON
        options={"temperature": 0.2},
    )
    return Form.model_validate_json(resp["message"]["content"])
