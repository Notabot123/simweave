## EdgeWeave

EdgeWeave is a flow-based low code tool, where execution paths are built using drawflow.js, and executed with python (fastAPI). It is an electron app, which both allows users to define their own nodes according to an SDK, and will also generate python code from graphs.

The SDK looks like this:

# sdk/decorators.py
```
from functools import wraps

NODE_REGISTRY: dict = {}    # name → full node metadata dict
BLOCK_SOURCES: dict = {}    # name → "core" | "project" | "user"

_LAYER_PRIORITY = {"core": 1, "project": 2, "user": 3}


def register_block(name: str, meta: dict):
    """
    Decorator that registers a node handler in NODE_REGISTRY.

    The ``meta`` dict may contain:

    - ``inputs``  – either an int count OR a list of ``{"name": ...}`` dicts.
      Both forms are accepted; a list is stored as-is so that the engine can
      use named-port resolution and the frontend can render port tooltips.
    - ``outputs`` – int count.
    - ``fields``  – list of UI field descriptors.
    - ``output_types`` – list of ``{"name": ..., "type": ...}`` dicts.
    - ``codegen`` – optional code-generation hints.
    - ``validate`` – optional validation rules list.
    """
    def decorator(func):
        raw_inputs = meta.get("inputs", 0)

        # Normalise: list of dicts → keep as-is; int → keep as int
        if isinstance(raw_inputs, list):
            inputs_meta = raw_inputs          # named-input list
            inputs_count = len(raw_inputs)
        else:
            inputs_meta = int(raw_inputs)     # legacy integer count
            inputs_count = int(raw_inputs)

        NODE_REGISTRY[name] = {
            "name": name,
            "label": meta.get("label", name),
            "category": meta.get("category", "Custom"),
            "icon": meta.get("icon"),
            # Full metadata list (or int for legacy nodes)
            "inputs": inputs_meta,
            # Convenience count used by the frontend port renderer
            "inputs_count": inputs_count,
            "outputs": meta.get("outputs", 1),
            "output_types": meta.get("output_types", []),
            # Both keys kept for backwards compat with existing frontend code
            "ui": meta.get("fields", []),
            "fields": meta.get("fields", []),
            "codegen": meta.get("codegen", {}),
            "validate": meta.get("validate", []),
            # When True the preview div is shown even when "node previews" is
            # toggled off in Settings (used by debug_print, Plotly nodes, etc.)
            "always_preview": bool(meta.get("always_preview", False)),
            "function": func,
        }

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator

```

# sdk/node.py
```
from functools import wraps
from sdk.decorators import register_block


def node(
    name,
    inputs=None,
    outputs=None,
    output_types=None,
    params=None,
    category="General",
    icon=None,
):
    """
    Decorator for defining a graph node using the high-level SDK.

    Parameters
    ----------
    name : str
        Unique block identifier (used as the registry key).
    inputs : list, optional
        Either a list of port-name strings (``["x", "y"]``) or a list of
        dicts with at least a ``"name"`` key and an optional ``"overrides"``
        key for the greyed-field feature
        (``[{"name": "start", "overrides": "start_value"}]``).
        An integer count is *not* accepted here; use the full list so that
        named-port metadata is preserved end-to-end.
    outputs : list, optional
        List of output port descriptors (currently used for port count).
    output_types : list, optional
        List of ``{"name": ..., "type": ...}`` dicts for connection colouring.
    params : dict, optional
        ``{param_name: default_value}`` pairs.  Types are inferred from the
        default values.
    category : str
        Sidebar category.
    icon : str, optional
        Emoji or short string shown on the node header.
    """

    inputs = inputs or []
    outputs = outputs or []
    output_types = output_types or []
    params = params or {}

    # Normalise inputs to a list of dicts with at least {"name": ...}
    normalised_inputs = []
    for inp in inputs:
        if isinstance(inp, str):
            normalised_inputs.append({"name": inp})
        elif isinstance(inp, dict):
            normalised_inputs.append(inp)
        # skip anything else gracefully

    def decorator(func):

        # ── Build UI field metadata from params ──────────────────────────────
        fields = []
        for k, v in params.items():
            if isinstance(v, bool):
                field_type = "checkbox"
            elif isinstance(v, float):
                field_type = "float"
            elif isinstance(v, int):
                field_type = "int"
            else:
                field_type = "text"

            fields.append({
                "type": field_type,
                "name": k,
                "label": k.replace("_", " ").title(),
                "default": v,
            })

        meta = {
            "category": category,
            "icon": icon,
            # Pass the full named-input list so the engine and frontend can
            # use port names and the "overrides" feature.
            "inputs": normalised_inputs,
            "outputs": max(1, len(outputs)),
            "output_types": output_types,
            "fields": fields,
        }

        # ── Engine-compatible wrapper ─────────────────────────────────────────
        @wraps(func)
        def handler(node_id, node_def, inputs, context, **kwargs):
            data = node_def.get("data", {}) or {}

            # Resolve params: prefer incoming dict (named-input mode) then
            # fall back to node data, then to the declared default.
            if isinstance(inputs, dict):
                params_dict = {
                    k: inputs.get(k) if inputs.get(k) is not None else data.get(k, v)
                    for k, v in params.items()
                }
                positional = list(inputs.values())
            else:
                params_dict = {k: data.get(k, v) for k, v in params.items()}
                positional = list(inputs) if inputs else []

            try:
                result = func(*positional, **params_dict)
            except Exception as e:
                return None, f"<pre>{e}</pre>"

            preview = None
            if hasattr(result, "head"):
                preview = result.head(20).to_html()

            return result, preview

        return register_block(name, meta)(handler)

    return decorator

```
This sim engine can be somewhat isolated from EdgeWeave, however there may be opportunities to integrate them tightly, for instance if cython was used or any compilation of this sim library that could be included in EdgeWeaves core. Please advise or document here, if you think of opportunities for tight integraton or justifications for not doing so.