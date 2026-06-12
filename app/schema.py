"""Form definition schema.

This is the contract the LLM must produce on every turn. Keep it small and
explicit — the model fills exactly these shapes, and we validate against them
before trusting anything. Later this maps onto the luke-capability-engine
form-definition format (formKey / versions); for now it is standalone.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field as PydField

FieldType = Literal[
    "text",
    "email",
    "number",
    "dropdown",
    "checkbox",
    "date",
    "textarea",
]


class Field(BaseModel):
    key: str = PydField(description="Stable machine key, snake_case, e.g. 'email_address'")
    label: str = PydField(description="Human-facing label shown next to the input")
    type: FieldType = "text"
    required: bool = False
    options: Optional[List[str]] = PydField(
        default=None, description="Choices for 'dropdown' fields; null otherwise"
    )
    placeholder: Optional[str] = None


class Form(BaseModel):
    title: str = "Untitled Form"
    fields: List[Field] = PydField(default_factory=list)


class ChatRequest(BaseModel):
    """One turn of the conversation. Stateless: the client sends the current
    form back each time, so no DB is needed yet."""
    message: str
    form: Optional[Form] = None


class ChatResponse(BaseModel):
    form: Form
    brain: str  # which LLM produced this ("gemini" | "ollama")
