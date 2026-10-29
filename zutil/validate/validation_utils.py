"""
Validation utilities for data preprocessing and common operations.
"""

from typing import Any, Dict

COLLECTION_KEYS = frozenset(
    {"fluid_zones", "boundary_conditions", "reference_conditions"}
)

REPORT_COLLECTION_KEYS = frozenset({"forces", "mass_flow", "monitor"})


def normalise_key(key: Any) -> Any:
    """The schema spelling of a dict key: spaces become underscores, in lower case."""
    return key.replace(" ", "_").lower() if isinstance(key, str) else key


def _report_labels(collection: dict) -> dict:
    """A report collection keyed by its deck labels, spaces underscored, refusing labels the solver reads as one."""
    labelled: Dict[Any, Any] = {}
    spellings: Dict[Any, Any] = {}
    for label, entry in collection.items():
        solver_label = normalise_key(label)
        if solver_label in spellings:
            raise ValueError(
                f"Duplicate report label: '{spellings[solver_label]}' and "
                f"'{label}' both write the report columns '{solver_label}_*' - "
                "give each report its own label."
            )
        spellings[solver_label] = label
        labelled[label.replace(" ", "_") if isinstance(label, str) else label] = (
            replace_spaces_in_keys(entry)
        )
    return labelled


def replace_spaces_in_keys(obj: Any, parent: Any = None) -> Any:
    """
    Recursively replace spaces with underscores and normalize keys.

    Two types of dictionaries are handled differently:
    1. Collection dictionaries: User-defined keys are preserved exactly as-is
    2. Model dictionaries: Schema-defined keys are normalized (spaces replaced, lowercase)

    Additionally, string values for the 'type' and 'kind' keys are lowercased
    to allow case-insensitive input (e.g., 'RANS' -> 'rans', 'LES' -> 'les').

    Collection dictionaries are ``COLLECTION_KEYS`` (fluid_zones,
    boundary_conditions, reference_conditions) wherever they appear, and
    ``REPORT_COLLECTION_KEYS`` (forces, mass_flow, monitor) directly under a
    ``report`` dict. Their keys are the deck author's labels. A report label
    is also the stem of its report columns, so it keeps its case but its
    spaces become underscores (``_report_labels``); ``to_cpp_dict`` then lower
    cases it, as it does every key, giving the column stem.

    Two differently-spelled keys that normalize to the same key within the
    same dictionary (e.g. 'convergence_control' and 'convergence control')
    raise ValueError rather than silently letting the later one win.

    Args:
        obj: Object to process (dict, list, or primitive)
        parent: The normalized key ``obj`` sits under, if any

    Returns:
        Processed object with appropriately normalized keys

    Raises:
        ValueError: if two differently-spelled keys in the same dictionary
            normalize to the same key, or two report labels in one
            collection differ only in case or spacing.
    """
    if isinstance(obj, dict):
        reports = REPORT_COLLECTION_KEYS if parent == "report" else frozenset()
        new_dict = {}
        seen_spellings: Dict[Any, Any] = {}
        for k, v in obj.items():
            normalized_key = normalise_key(k)

            if normalized_key in seen_spellings:
                raise ValueError(
                    f"Duplicate key: '{seen_spellings[normalized_key]}' and "
                    f"'{k}' are the same setting spelt two ways. Use one "
                    "spelling."
                )
            seen_spellings[normalized_key] = k

            if normalized_key in reports and isinstance(v, dict):
                new_dict[normalized_key] = _report_labels(v)
            elif normalized_key in COLLECTION_KEYS and isinstance(v, dict):
                new_dict[normalized_key] = {
                    label: replace_spaces_in_keys(entry) for label, entry in v.items()
                }
            else:
                processed_value = replace_spaces_in_keys(v, normalized_key)
                if (
                    normalized_key in ("type", "kind")
                    and isinstance(processed_value, str)
                    and not processed_value.startswith("DECIDE:")
                ):
                    processed_value = processed_value.lower()

                new_dict[normalized_key] = processed_value
        return new_dict
    elif isinstance(obj, list):
        return [replace_spaces_in_keys(item) for item in obj]
    else:
        return obj


def expand_reference_inheritance(
    data: Dict[str, Any], reference_field: str = "reference"
) -> Dict[str, Any]:
    """
    Expand inheritance in data where items use 'reference': 'base_item_name'
    to inherit fields from a base item and then override/add specific fields.

    Args:
        data: Dictionary with potential reference inheritance
        reference_field: Name of the field that indicates inheritance

    Returns:
        Dictionary with inheritance expanded

    Example:
        Input:
        {
            "initial": {"temperature": 300, "pressure": 101325},
            "inflow": {"reference": "initial", "velocity": 10},
            "outflow": {"reference": "initial", "pressure": 90000}
        }

        Output:
        {
            "initial": {"temperature": 300, "pressure": 101325},
            "inflow": {"temperature": 300, "pressure": 101325, "velocity": 10},
            "outflow": {"temperature": 300, "pressure": 90000}  # pressure overridden
        }
    """
    expanded_data = {}

    # First pass: copy all items without references
    for key, item_data in data.items():
        if isinstance(item_data, dict) and reference_field not in item_data:
            expanded_data[key] = item_data.copy()

    # Second pass: expand items with references, in dependency order.
    #
    # A single sweep only resolves a reference chain (e.g. C -> B -> A)
    # when the base is expanded before the item that depends on it, which
    # made resolution order-dependent on declaration order in the source
    # dict. Instead, repeatedly sweep the still-pending referencing items
    # and expand whichever ones have their base already expanded, until a
    # sweep makes no further progress - a fixed-point iteration that
    # resolves chains of any length regardless of declaration order.
    pending = {
        key: item_data
        for key, item_data in data.items()
        if isinstance(item_data, dict) and reference_field in item_data
    }

    while pending:
        resolved_this_sweep = []
        for key, item_data in pending.items():
            base_ref_name = item_data[reference_field]

            # Base not expanded yet (possibly still pending itself) - try
            # again on a later sweep.
            if base_ref_name not in expanded_data:
                continue

            # Start with a copy of the base item
            expanded_item = expanded_data[base_ref_name].copy()

            # Deep copy nested dictionaries to avoid mutation
            for field_key, field_value in expanded_item.items():
                if isinstance(field_value, dict):
                    expanded_item[field_key] = field_value.copy()

            # Override/add fields from the current item (excluding reference field)
            for field_key, field_value in item_data.items():
                if field_key != reference_field:
                    expanded_item[field_key] = field_value

            expanded_data[key] = expanded_item
            resolved_this_sweep.append(key)

        if not resolved_this_sweep:
            # Nothing resolvable left this sweep. Distinguish a genuinely
            # undefined base (not present anywhere in the input) - keeping
            # the original wording for that case - from a base that IS
            # present but never resolves, which means the remaining
            # pending items form a reference cycle.
            for key, item_data in pending.items():
                base_ref_name = item_data[reference_field]
                if base_ref_name not in data:
                    raise ValueError(
                        f"Item '{key}' references '{base_ref_name}' which is "
                        "not defined or not yet expanded"
                    )

            unresolved = ", ".join(
                f"'{key}' -> '{item_data[reference_field]}'"
                for key, item_data in pending.items()
            )
            raise ValueError(f"Circular reference chain detected: {unresolved}")

        for key in resolved_this_sweep:
            del pending[key]

    # Third pass: anything left over (non-dict entries) that the first two
    # passes didn't copy.
    for key, item_data in data.items():
        if key not in expanded_data:
            if isinstance(item_data, dict):
                expanded_data[key] = item_data.copy()
            elif hasattr(item_data, "model_copy"):
                expanded_data[key] = item_data.model_copy()
            elif hasattr(item_data, "copy"):
                expanded_data[key] = item_data.copy()
            else:
                expanded_data[key] = item_data

    return expanded_data


def lowercase_keys_recursive(d: dict) -> dict:
    """Return a copy of *d* with all string keys lowercased, recursively.

    The C++ ``PythonDictParameters`` methods (``getDict``, ``getString``,
    etc.) call ``boost::algorithm::to_lower`` on every key before look-up,
    so the Python dict must use lowercase keys to match.
    """
    out = {}
    for key, val in d.items():
        lk = key.lower() if isinstance(key, str) else key
        if isinstance(val, dict):
            out[lk] = lowercase_keys_recursive(val)
        elif isinstance(val, list):
            out[lk] = [
                lowercase_keys_recursive(v) if isinstance(v, dict) else v for v in val
            ]
        else:
            out[lk] = val
    return out


def force_reports(parameters: dict, model_name: str) -> dict:
    """The force reports the solver writes for ``model_name``.

    ``parameters`` is a resolved deck, so a model block's ``output_settings``
    already holds the reports it inherits from the solver block.
    """
    model = (parameters.get("model") or {}).get(model_name) or {}
    block = model if "output_settings" in model else parameters.get("solver") or {}
    report = (block.get("output_settings") or {}).get("report") or {}
    return report.get("forces") or {}
