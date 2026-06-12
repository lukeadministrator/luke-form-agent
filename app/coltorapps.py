"""Convert between the LLM's flat FormSpec and the coltorapps builder schema.

coltorapps schema shape (what luke-consumer-ui's FormRenderer / builderStore and
luke-capability-engine's draftSchema expect):

    {
      "entities": {
        "<id>": { "type": "textField", "attributes": { "label": ..., "key": ..., "required": ... } },
        ...
      },
      "root": ["<id>", "<id>", ...]   # top-level field order
    }

The submission key for a field is `attributes.key` (falling back to the entity
id). We always set `key` explicitly so data keys are stable across edits.
"""
from __future__ import annotations

import uuid

from .schema import CHOICE_TYPES, FieldType, FormSpec, SpecField

# coltorapps field types we understand on the way IN (schema -> spec). Anything
# else (containers, content, layout) is left untouched and preserved verbatim.
KNOWN_FIELD_TYPES = {
    "textField", "textarea", "number", "email", "phoneNumber",
    "checkbox", "select", "radio", "selectBoxes", "datetime", "currency",
}


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def schema_to_spec(schema: dict | None) -> tuple[FormSpec, dict, dict, list]:
    """Project an incoming coltorapps schema down to a flat FormSpec the LLM can
    edit. Also return bookkeeping needed to rebuild without losing data:

      - existing: {key -> {"id", "type", "attributes"}} for known simple fields,
        so edits can merge onto (preserve) the original attributes.
      - preserved_entities: {id -> entity} for entities we DON'T expose to the
        LLM (containers/content/unknown) — carried through unchanged.
      - preserved_root_ids: their ids, kept so they survive in `root`.
    """
    schema = schema or {}
    entities = schema.get("entities", {}) or {}
    root = schema.get("root") or list(entities.keys())

    fields: list[SpecField] = []
    existing: dict = {}
    preserved_entities: dict = {}
    preserved_root_ids: list = []

    for eid in root:
        ent = entities.get(eid)
        if not isinstance(ent, dict):
            continue
        etype = ent.get("type", "")
        attrs = ent.get("attributes", {}) or {}
        if etype in KNOWN_FIELD_TYPES:
            key = str(attrs.get("key") or eid)
            opts = attrs.get("options")
            fields.append(
                SpecField(
                    key=key,
                    label=str(attrs.get("label", key)),
                    type=etype,  # type: ignore[arg-type]
                    required=bool(attrs.get("required", False)),
                    options=list(opts) if isinstance(opts, list) else None,
                    placeholder=attrs.get("placeholder"),
                )
            )
            existing[key] = {"id": eid, "type": etype, "attributes": dict(attrs)}
        else:
            # Container / content / unknown — preserve as-is.
            preserved_entities[eid] = ent
            preserved_root_ids.append(eid)

    return FormSpec(fields=fields), existing, preserved_entities, preserved_root_ids


def spec_to_schema(
    spec: FormSpec,
    existing: dict | None = None,
    preserved_entities: dict | None = None,
    preserved_root_ids: list | None = None,
) -> dict:
    """Render a FormSpec into a coltorapps schema. For fields whose key matches
    an existing entity, reuse its id and merge onto its attributes so advanced
    builder settings (logic, validation, etc.) survive. Preserved non-field
    entities are carried through and appended after the simple fields."""
    existing = existing or {}
    preserved_entities = dict(preserved_entities or {})
    preserved_root_ids = list(preserved_root_ids or [])

    entities: dict = {}
    root: list = []

    for f in spec.fields:
        prev = existing.get(f.key)
        eid = prev["id"] if prev else _new_id()
        attrs = dict(prev["attributes"]) if prev else {}

        attrs["label"] = f.label
        attrs["key"] = f.key
        attrs["required"] = f.required
        if f.placeholder is not None:
            attrs["placeholder"] = f.placeholder

        if f.type in CHOICE_TYPES:
            attrs["options"] = f.options or ["Option 1"]
        else:
            attrs.pop("options", None)

        entities[eid] = {"type": f.type, "attributes": attrs}
        root.append(eid)

    # Carry preserved entities through unchanged.
    for eid in preserved_root_ids:
        if eid in preserved_entities:
            entities[eid] = preserved_entities[eid]
            root.append(eid)

    return {"entities": entities, "root": root}
