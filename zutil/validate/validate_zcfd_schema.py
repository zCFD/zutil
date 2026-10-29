"""
Validate a zCFD deck: one pydantic pass, then one mapping of its errors to nodes.

  - validate_zcfd_comprehensive() — returns (model_or_none, nodes)
"""

import logging

from pydantic import ValidationError

from .error_mapping import _clean_loc_parts, map_errors_to_nodes
from .validation_utils import replace_spaces_in_keys
from .validation_utils import normalise_key
from .zcfd_schema import (
    InheritedSettings,
    ZCFDInputModelBase,
    collecting_warnings,
    inherited_settings,
    resolve_mesh_settings,
)

_logger = logging.getLogger("zutil.validate")


def _pathless(message: str) -> list:
    return map_errors_to_nodes([{"type": "deck", "loc": (), "msg": message}])


def _repeats_solver(entries: list, inherited: InheritedSettings) -> set:
    """Indices of the ``(loc, msg)`` entries a mesh inherited along with the solver's own."""
    at_solver = {
        (tuple(map(normalise_key, loc)), msg)
        for loc, msg in entries
        if loc[:1] == ("solver",)
    }
    return {
        i
        for i, (loc, msg) in enumerate(entries)
        if loc[:1] == ("model",) and (inherited.source(loc), msg) in at_solver
    }


def _open_decisions(node, path=()) -> list:
    """An error for each DECIDE placeholder update_input left in the deck."""
    if isinstance(node, str):
        if not node.startswith("DECIDE:") or not path:
            return []
        return [
            {
                "type": "open_decision",
                "loc": path,
                "msg": f"Open decision left by update_input: {node}",
            }
        ]
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, (list, tuple)):
        items = enumerate(node)
    else:
        return []
    return [e for k, v in items for e in _open_decisions(v, path + (k,))]


def validate_zcfd_comprehensive(data: dict) -> tuple:
    """Validate a zCFD configuration dict, reporting every error at once.

    Keys are first normalised (spaces to underscores, ``type`` and ``kind``
    values lowercased), then the deck is validated once against
    :class:`ZCFDInputModelBase`.

    Returns:
        (validated_model, [])                     if valid, no warnings
        (validated_model, [warning_node, ...])     if valid, with warnings
        (None, [error_node, ...])                  if invalid (mixed with
                                                     any warning_nodes too)

    Each node has:
        nodes: list[str]        — deprecated: zui tree node names; read
                                   field_path instead
        field_name: str         — leaf field for panel highlighting
        field_path: str         — root-anchored path in schema spelling
        title: str              — the field's schema title
        display_message: str    — human-readable tooltip text
        message: str            — ``field_path: display_message``
        anchor: str | None      — reference ``route#fragment`` for the
                                   field, None when the error has no path
        severity: str           — "error" or "warning". A config is only
                                   invalid if at least one node is an error.
    """
    if not isinstance(data, dict):
        return None, _pathless("Input data is None (failed to load parameters?)")
    try:
        deck = replace_spaces_in_keys(data)
    except ValueError as e:
        return None, _pathless(str(e))

    decisions = _open_decisions(deck)
    decided = {tuple(map(str, d["loc"])) for d in decisions}
    inherited = inherited_settings(deck)
    deck = resolve_mesh_settings(deck)
    with collecting_warnings() as warnings:
        try:
            model = ZCFDInputModelBase.model_validate(deck)
        except ValidationError as e:
            raised = e.errors()
            locs = [tuple(_clean_loc_parts(err["loc"], deck)) for err in raised]
            repeated = _repeats_solver(
                [(loc, err["msg"]) for loc, err in zip(locs, raised)], inherited
            )
            errors = [
                err
                for i, (loc, err) in enumerate(zip(locs, raised))
                if i not in repeated and tuple(map(str, loc)) not in decided
            ]
            model, errors = None, map_errors_to_nodes(decisions + errors, deck)
        except Exception:
            _logger.exception("Unexpected error while validating the deck.")
            model, errors = (
                None,
                _pathless("Internal error during validation — see logs for details."),
            )
        else:
            errors = map_errors_to_nodes(decisions, deck)
            if errors:
                model = None
    repeated = _repeats_solver(
        [(tuple(loc), text) for loc, text in warnings], inherited
    )
    warned = map_errors_to_nodes(
        [
            {"type": "deck_warning", "loc": loc, "msg": text}
            for i, (loc, text) in enumerate(warnings)
            if i not in repeated
        ],
        deck,
        severity="warning",
    )
    return model, errors + warned


__all__ = [
    "validate_zcfd_comprehensive",
]
