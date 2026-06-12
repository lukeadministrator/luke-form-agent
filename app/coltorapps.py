"""Convert between the LLM's flat FormSpec and the coltorapps builder schema.

coltorapps schema shape (what luke-consumer-ui's FormRenderer / builderStore and
luke-capability-engine's draftSchema expect):

    {
      "entities": {
        "<id>": { "type": "textField", "attributes": { "label": ..., "key": ..., "required": ... } },
        "<id>": { "type": "panel", "attributes": {...}, "children": ["<id>", ...] },
        ...
      },
      "root": ["<id>", "<id>", ...]   # top-level order
    }

The submission key for a field is `attributes.key` (falling back to the entity
id). We always set `key` explicitly so data keys are stable across edits.

We only expose flat, top-level "simple" fields to the LLM. Everything else
(containers, their nested children, and advanced field types we don't model) is
preserved verbatim so AI edits never corrupt a complex form. Preserved
top-level entities are kept after the simple fields in `root`.
"""
from __future__ import annotations

import uuid

from .schema import CHOICE_TYPES, FormSpec, SpecField

# coltorapps leaf field types we let the LLM build/edit. Anything else
# (containers, content, file/signature/tags/etc.) is preserved untouched.
# Every one of these spreads the `api` attribute group, so `key` is always valid.
KNOWN_FIELD_TYPES = {
    "textField", "textarea", "number", "email", "phoneNumber",
    "checkbox", "select", "radio", "selectBoxes", "datetime", "currency",
}

# Types whose coltorapps definition includes `placeholderAttribute`. Setting
# `placeholder` on any other type produces an "Unknown entity attribute" schema.
PLACEHOLDER_TYPES = {
    "textField", "textarea", "number", "email", "phoneNumber", "currency", "select",
}


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def schema_to_spec(schema: dict | None) -> tuple[FormSpec, dict, dict, list]:
    """Project a coltorapps schema down to a flat FormSpec the LLM can edit, plus
    the bookkeeping needed to rebuild without losing anything:

      - existing: {key -> {"id","type","attributes"}} for simple fields, so edits
        merge onto (preserve) the original attributes / entity id.
      - preserved_entities: {id -> entity} for EVERY entity that isn't a rebuilt
        simple field — containers AND their nested children — carried verbatim.
      - preserved_root_ids: the top-level ids among those, kept for `root`.
    """
    schema = schema or {}
    entities = schema.get("entities", {}) or {}
    root = schema.get("root") or list(entities.keys())

    fields: list[SpecField] = []
    existing: dict = {}
    simple_ids: set = set()
    preserved_root_ids: list = []

    for eid in root:
        ent = entities.get(eid)
        if not isinstance(ent, dict):
            continue
        etype = ent.get("type", "")
        attrs = ent.get("attributes", {}) or {}
        is_leaf = not ent.get("children")
        if etype in KNOWN_FIELD_TYPES and is_leaf:
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
            simple_ids.add(eid)
        else:
            preserved_root_ids.append(eid)

    # Preserve every entity that isn't a rebuilt simple field — this includes
    # containers AND their nested children (which never appear in `root`).
    preserved_entities = {
        eid: ent for eid, ent in entities.items() if eid not in simple_ids
    }

    return FormSpec(fields=fields), existing, preserved_entities, preserved_root_ids


def spec_to_schema(
    spec: FormSpec,
    existing: dict | None = None,
    preserved_entities: dict | None = None,
    preserved_root_ids: list | None = None,
) -> dict:
    """Render a FormSpec into a coltorapps schema. Fields whose key matches an
    existing entity reuse its id and merge onto its attributes (so advanced
    builder settings survive). Preserved entities are carried through verbatim;
    preserved top-level ones are appended after the simple fields in `root`."""
    existing = existing or {}
    preserved_entities = dict(preserved_entities or {})
    preserved_root_ids = list(preserved_root_ids or [])

    entities: dict = {}
    root: list = []
    used: set = set()

    for f in spec.fields:
        prev = existing.get(f.key)
        eid = prev["id"] if prev else _new_id()
        # Never collide with a preserved entity id or one already emitted.
        while eid in preserved_entities or eid in used:
            eid = _new_id()
        used.add(eid)

        # Merge onto the prior attributes ONLY when the type is unchanged — a
        # different type has a different (incompatible) attribute set, so reusing
        # e.g. textField's minLength on a select would be an invalid attribute.
        same_type = bool(prev) and prev.get("type") == f.type
        attrs = dict(prev["attributes"]) if same_type else {}

        attrs["label"] = f.label
        attrs["key"] = f.key
        attrs["required"] = f.required

        if f.type in PLACEHOLDER_TYPES and f.placeholder is not None:
            attrs["placeholder"] = f.placeholder
        elif f.type not in PLACEHOLDER_TYPES:
            attrs.pop("placeholder", None)  # strip if carried from a prior type

        if f.type in CHOICE_TYPES:
            attrs["options"] = f.options or ["Option 1"]
        else:
            attrs.pop("options", None)

        entities[eid] = {"type": f.type, "attributes": attrs}
        root.append(eid)

    # Carry every preserved entity through unchanged (containers + their children).
    for eid, ent in preserved_entities.items():
        entities[eid] = ent
    # Keep preserved top-level entities in `root`, after the simple fields.
    for eid in preserved_root_ids:
        if eid in preserved_entities:
            root.append(eid)

    return {"entities": entities, "root": root}
