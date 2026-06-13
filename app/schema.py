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
    "button",
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


class AssistantTurn(FormSpec):
    """What the LLM returns each turn: the complete form PLUS a conversational
    reply and a few suggested next steps the user can act on."""
    reply: str = PydField(
        default="",
        description="Friendly, first-person natural-language reply describing what you did "
        "(or a clarifying question). 1-3 sentences, conversational.",
    )
    suggestions: List[str] = PydField(
        default_factory=list,
        description="2-4 short, actionable next-step ideas as imperatives, e.g. "
        "'Add a phone number'. Each under ~6 words.",
    )


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
    reply: str = ""  # natural-language message to show the user
    suggestions: List[str] = []  # clickable next-step ideas
    changed: bool = True  # False when the form was untouched (e.g. a question)
    brain: str  # which LLM produced this ("groq" | "gemini" | "ollama")
