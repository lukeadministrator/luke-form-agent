"""Form spec schema — the LLM's contract.

The LLM produces a flat, easy-to-get-right field list (`FormSpec`). Python then
deterministically renders that into the coltorapps builder schema that
luke-consumer-ui / luke-capability-engine actually consume (see coltorapps.py).

Field types are the coltorapps palette names so the mapping is loss-free.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field as PydField

# coltorapps palette types we support generating. Choice types (select/radio/
# selectBoxes) require an `options` list.
FieldType = Literal[
    "textField",
    "textarea",
    "number",
    "email",
    "phoneNumber",
    "checkbox",
    "select",
    "radio",
    "selectBoxes",
    "datetime",
    "currency",
]

CHOICE_TYPES = {"select", "radio", "selectBoxes"}


class SpecField(BaseModel):
    key: str = PydField(description="Stable snake_case data key, e.g. 'email_address'")
    label: str = PydField(description="Human-facing label")
    type: FieldType = "textField"
    required: bool = False
    options: Optional[List[str]] = PydField(
        default=None, description="Choices for select/radio/selectBoxes; null otherwise"
    )
    placeholder: Optional[str] = None


class FormSpec(BaseModel):
    title: str = "Untitled Form"
    fields: List[SpecField] = PydField(default_factory=list)


class ChatRequest(BaseModel):
    """One turn. Stateless: the client (the Form Builder) sends the current
    coltorapps schema back each time, so the agent needs no storage."""
    message: str
    # Current coltorapps schema: {"entities": {...}, "root": [...]}. Optional /
    # empty for a brand-new form.
    schema: Optional[dict] = None
    title: Optional[str] = None  # current form name, if the client tracks one


class ChatResponse(BaseModel):
    schema: dict  # updated coltorapps schema, ready for builderStore / saveDraft
    title: str
    brain: str  # which LLM produced this ("groq" | "gemini" | "ollama")
