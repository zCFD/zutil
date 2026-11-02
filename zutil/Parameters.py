"""
Copyright (c) 2012-2026, Zenotech Ltd
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Zenotech Ltd nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ZENOTECH LTD BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import logging
from typing import Optional

import numpy as np

from zutil.fileutils import get_parameters_from_file
from zutil.deck_dependencies import (
    DeckDependencies,
    check_for_deck_change,
    invalidate_cached_modules,
    track_dependencies,
)

# Initialize logger using standard logging
logger = logging.getLogger(__name__)

# Timestamp of parameter file when file was last read
last_timestamp: float = 0.0


def _compare_nested_structures(s1, s2, returned_keys: dict, levels: str = "") -> None:
    """Iterate through nested structures (dicts, lists, arrays) and return a dictionary of changed values."""
    if isinstance(s1, dict) and isinstance(s2, dict):
        for key, val in s1.items():
            if key in s2:
                _compare_nested_structures(
                    val, s2[key], returned_keys, levels + key + ":"
                )
            else:
                k = levels + key if levels else key
                returned_keys[k] = (val, "Not previously set")
    elif isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray):
        if not np.array_equal(s1, s2):
            k = levels.rstrip(":")
            returned_keys[k] = (s1, s2)
    elif isinstance(s1, (list, tuple)) and isinstance(s2, (list, tuple)):
        if len(s1) != len(s2) or any(v1 != v2 for v1, v2 in zip(s1, s2)):
            k = levels.rstrip(":")
            returned_keys[k] = (s1, s2)
    elif callable(s1):
        pass
    else:
        if s1 != s2:
            k = levels.rstrip(":")
            returned_keys[k] = (s1, s2)


class DynamicUpdate:
    """The outcome of one live-reload poll that saw the deck change.

    ``accepted`` and ``rejected`` are keyed by the diff key exactly as
    ``_compare_nested_structures`` spelled it — mesh-name segment and all —
    so an accepted per-mesh override still says which mesh it belongs to.
    Values are the ``(new, old)`` pairs from the diff. ``reasons`` gives
    one sentence per rejected key saying why it was not applied.
    ``echoes`` are the mesh keys whose change a mesh inherited from the same
    change to the solver block; they apply, but the readout shows only the
    solver's.

    ``files`` are the deck files whose edit triggered the poll. ``error``
    is set when the edited deck could not be read or failed validation; in
    that case both mappings are empty and the run keeps its parameters.

    Rejection is per-key: a change to a non-updatable field discards only
    that key, leaving the rest to be applied. v1 discarded the whole diff.
    """

    NOT_UPDATABLE = "not changeable mid-run"
    NOT_TRANSFERABLE = "a function defined in the deck cannot be shared across ranks"

    def __init__(
        self,
        accepted: dict,
        rejected: dict,
        reasons: Optional[dict] = None,
        files: Optional[list] = None,
        error: Optional[str] = None,
        echoes: Optional[set] = None,
    ) -> None:
        self.accepted = dict(accepted)
        self.rejected = dict(rejected)
        self.reasons = {
            key: (reasons or {}).get(key, self.NOT_UPDATABLE) for key in self.rejected
        }
        self.files = list(files or [])
        self.error = error
        self.echoes = set(echoes or ())

    @classmethod
    def failed(cls, files: list, error: str) -> "DynamicUpdate":
        """A poll whose re-read did not produce a usable deck."""
        return cls({}, {}, files=files, error=error)

    def transferable(self) -> "DynamicUpdate":
        """A copy safe to send through ``comm.bcast``.

        Rank 0 broadcasts the partition so every rank applies the same
        change. Anything a deck defines inside itself — a force-driver
        transform, a ramp function — cannot be pickled by qualified name,
        because the deck is ``exec``'d rather than imported, so a value
        carrying one would take the run down at the broadcast rather than
        at any point that could be diagnosed from the deck.

        An accepted entry whose value will not pickle is moved to rejected
        — it cannot be applied consistently across ranks, and applying it
        on rank 0 alone would put the ranks into different states, which is
        worse than not applying it. A rejected entry keeps its values when
        they pickle (so the readout can show what was ignored) and is
        reduced to ``(None, None)`` otherwise.
        """
        import pickle

        def _pickles(value: object) -> bool:
            try:
                pickle.dumps(value)
            except Exception:
                return False
            return True

        accepted = {}
        rejected = {}
        reasons = dict(self.reasons)
        for key, change in self.rejected.items():
            rejected[key] = change if _pickles(change) else (None, None)
        for key, change in self.accepted.items():
            if _pickles(change):
                accepted[key] = change
            else:
                rejected[key] = (None, None)
                reasons[key] = self.NOT_TRANSFERABLE
        return DynamicUpdate(
            accepted, rejected, reasons, self.files, self.error, self.echoes
        )

    def apply_to(self, params: dict) -> None:
        """Write the accepted changes into ``params`` in place.

        Merging rather than swapping in the freshly-read dict matters: the
        running configuration has been through post-validation steps the
        re-read has not (force-driver callables installed into reference
        conditions, mesh-scale folding, restart-name resolution). Replacing
        it wholesale would silently drop those, so only the keys that passed
        the whitelist gate are written, and everything else keeps the object
        the solve started with.

        Diff keys are literal dict paths — that is what produced them — so
        no schema normalisation is needed here; the mesh-name segment of a
        per-mesh override is exactly the dict key it must be written under.

        Nothing is copied, not even the dicts along a written path. C++ was
        given these dict objects at ``reset_dictionary`` and reads values
        such as the output frequencies from them on every call, so a write
        into a copy would be logged as applied while C++ kept the old value.
        A path missing from ``params`` is created as new dicts.
        """
        for key, (new_value, _old) in self.accepted.items():
            segments = key.split(":") if ":" in key else key.split(".")
            node = params
            for segment in segments[:-1]:
                child = node.get(segment)
                if not isinstance(child, dict):
                    child = {}
                    node[segment] = child
                node = child
            node[segments[-1]] = new_value

    def __repr__(self) -> str:
        return (
            f"DynamicUpdate({len(self.accepted)} accepted, "
            f"{len(self.rejected)} rejected)"
        )


class Parameters:
    def read(self, configfile: str, controlfile: str) -> dict:
        """The deck's ``parameters`` dict as written, keys not yet normalised.

        Records ``controlfile``'s mtime for :meth:`read_if_changed`.
        """
        logger.debug(__file__ + " " + __name__ + ":read")

        if not os.path.isabs(configfile):
            configfile = os.path.abspath(configfile)

        parameters = get_parameters_from_file(configfile)

        props = os.stat(controlfile)
        global last_timestamp
        last_timestamp = props.st_mtime
        return dict({}, **parameters)

    def validate_parameters(
        self, unvalidated_parameters: dict, origin: Optional[str] = None
    ) -> tuple:
        """Validate a deck read by :meth:`read`, logging what the engineer must see.

        ``origin`` is the deck file, which names it in routing messages.
        Each warning is logged on the ``zutil.validate`` logger and any
        error text on this module's logger; the launcher routes both to
        stderr on rank 0 only.

        Returns:
            ``(True, {"_validated_model": model})`` on success.
            ``(False, message)`` on failure, where ``message`` is the
            formatted error text.
        """
        from zutil.validate.deck import check_deck

        report = check_deck(unvalidated_parameters, origin=origin)
        report.log_warnings()
        if not report.ok:
            message = report.text()
            logger.error(message)
            return False, message
        return True, {"_validated_model": report.model}

    def partition_dynamic_update(self, keys: dict) -> DynamicUpdate:
        """Split a parameter-dict diff into live-applicable and rejected keys.

        Enforcement is driven by the ``updatable=True`` metadata that schema
        authors attach via ``zCFD_Field`` (see
        ``zutil.validate.schemas.base_types.zCFD_Field``); the whitelist of
        allowed paths is computed by
        ``zutil.validate.updatable.updatable_paths``.

        Rejection is per-key rather than v1's all-or-nothing: a diff that
        touches one non-updatable field still applies everything else, and
        only the offending keys are dropped. Nothing is logged here: the
        caller reports the whole outcome in one place.

        Args:
            keys: mapping of changed-parameter path -> (new_value,
                old_value), as produced by ``_compare_nested_structures``.
                Paths may be ``:``-joined (the convention that function
                uses) or ``.``-joined (the schema attribute-path
                convention), and a per-mesh override may carry its mesh
                name as a segment; all three are normalised before matching.

        Returns:
            A :class:`DynamicUpdate` holding the accepted and rejected
            subsets of ``keys``.
        """
        from zutil.validate.updatable import is_updatable

        accepted = {}
        rejected = {}
        for key, change in keys.items():
            if is_updatable(key):
                accepted[key] = change
            else:
                rejected[key] = change

        return DynamicUpdate(accepted, rejected)

    def read_if_changed(
        self, old_params: dict, case_name: str, control_file: str
    ) -> tuple:
        global last_timestamp
        props = os.stat(control_file)
        if props.st_mtime > last_timestamp:
            params = self.read(case_name, control_file)
            return True, params
        return False, old_params


class ParameterHelper:
    """Accessor for solver parameters with model→solver fallback.

    Mirrors the C++ Parameters.h hierarchical lookup: for overrideable
    fields the lookup order is model[mesh_name].<field> first, then
    solver.<field> as fallback.

    Args:
        parameters: The full validated parameter dict.
        model_name: The mesh/model name for model-block lookups.
    """

    def __init__(self, parameters: dict, model_name: str) -> None:
        self._params = parameters
        self._model_name = model_name
        self._model_block = parameters.get("model", {}).get(model_name, {})
        self._solver_block = parameters.get("solver", {})

    # ------------------------------------------------------------------
    # Core accessor
    # ------------------------------------------------------------------

    def get(self, field: str, default=None):
        """Return *field* with model→solver fallback for overrideables.

        For non-overrideable fields, only the model block is checked.
        """
        from zutil.validate.schemas.solver_schemas import MESH_OVERRIDES

        if field in MESH_OVERRIDES:
            val = self._model_block.get(field)
            if val is not None:
                return val
            return self._solver_block.get(field, default)
        return self._model_block.get(field, default)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def convergence_control(self) -> dict:
        return self.get("convergence_control", {})

    @property
    def solver_convergence_control(self) -> dict:
        """Solver-level convergence_control, ignoring any per-mesh override.

        Mirrors ``ParameterHelper::solverConvergenceControl`` in
        ``src/util/Parameters.h``: for the keys that are global to the run by
        construction — the incompressible coupling scheme and its relaxation —
        a mesh block is forbidden to carry them, so the override lookup would
        find nothing.
        """
        return self._solver_block.get("convergence_control", {}) or {}

    @property
    def numerical_scheme(self) -> dict:
        return self.get("numerical_scheme", {})

    @property
    def output_settings(self) -> dict:
        return self.get("output_settings", {})

    @property
    def initialisation(self) -> dict:
        return self.get("initialisation", {})

    @property
    def time_settings(self) -> dict:
        """time_settings lives only on the solver block."""
        return self._solver_block.get("time_settings", {})

    @property
    def model_block(self) -> dict:
        """Direct access to the raw model block dict."""
        return self._model_block

    @property
    def solver_block(self) -> dict:
        """Direct access to the raw solver block dict."""
        return self._solver_block

    @property
    def parameters(self) -> dict:
        """The full parameter dict."""
        return self._params


class DeckReloadError(RuntimeError):
    """A changed deck could not be re-read or failed validation."""


class DeckReloader:
    """Detects live edits to a case's deck and reports what may be applied.

    The v2 replacement for v1's ``check_for_control_dict_change``, assembled
    from the pieces this feature is built on: dependency-set change
    detection, the rank-0-read-and-broadcast wrapper, re-validation through
    the normal pydantic path, the ``_compare_nested_structures`` diff, and
    the per-key whitelist gate.

    Reports only — applying the accepted changes to the running solver is
    the caller's job, so this stays free of solver imports and is testable
    without one.

    Usage::

        reloader = DeckReloader(case_name, control_file)
        reloader.prime()
        ...
        update = reloader.poll(comm)      # at report cadence
        if update is not None:
            apply(update.accepted)

    **The baseline is the validated dict**, not the raw deck. Diffing
    ``to_cpp_dict()`` output on both sides means the diff keys are already
    schema paths (``solver:convergence_control:cfl:cfl``), which is exactly
    what the whitelist matches on, and a key *deleted* from the deck shows
    up as its schema default reappearing rather than vanishing unnoticed.
    The launcher's post-validation steps (mesh-scale folding, force-driver
    installation, restart-name resolution) are deliberately not re-run: both
    sides of the diff are pre-post-processing, so the comparison stays
    apples-to-apples, and this module stays free of any ``zcfd`` import.

    Rank 0 owns the filesystem work, the diff, and the baseline. Only the
    resulting partition is broadcast — never the parameter dict itself,
    which holds the deck's own callables (force-driver transforms, ramp
    functions) and cannot be pickled: a deck is ``exec``'d, not imported,
    so nothing defined inside one is reachable by qualified name. Other
    ranks never diff, so they never need a baseline; they take the accepted
    changes and merge them into the configuration they are already running.
    """

    def __init__(self, case_name: str, control_file: str) -> None:
        self.case_name = case_name
        self.control_file = control_file
        self.dependencies = DeckDependencies()
        self.baseline: dict = {}
        self.inherited = None

    def prime(self) -> dict:
        """Read the deck and take the result as the baseline to diff against.

        Called once before the solve starts. Returns the validated dict.
        Warnings are not logged again: the launcher has just logged them.
        """
        from zutil.validate.zcfd_schema import inherited_settings

        report, dependencies = self._read()
        self.baseline = report.model.to_cpp_dict()
        self.inherited = inherited_settings(report.parameters)
        self.dependencies = dependencies
        return self.baseline

    def poll(self, comm=None) -> Optional[DynamicUpdate]:
        """Check for an edit and return what happened, or ``None`` if nothing did.

        ``None`` means the deck is unchanged. Otherwise the returned
        :class:`DynamicUpdate` names the edited files and carries one of:
        the accepted/rejected partition (either may be empty — a save with
        no effective change has both empty), or ``error`` when the deck
        could not be re-read or failed validation. Nothing is logged here;
        the caller owns the readout.
        """
        result = check_for_deck_change(
            self.dependencies, comm=comm, on_change=self._reread_and_diff
        )
        if not result.changed:
            return None
        if result.failed or result.payload is None:
            return DynamicUpdate.failed(result.changed, result.error or "unknown error")
        update = result.payload
        update.files = list(result.changed)
        return update

    def _read(self) -> tuple:
        """Read and check the deck; return ``(report, dependencies)``.

        Raises :class:`DeckReloadError` carrying the error text when the
        deck has errors.
        """
        from zutil.validate.deck import check_deck

        with track_dependencies() as dependencies:
            unvalidated = Parameters().read(self.case_name, self.control_file)

        report = check_deck(unvalidated, origin=self.control_file)
        if not report.ok:
            raise DeckReloadError(report.text())
        return report, dependencies

    def _reread_and_diff(self) -> DynamicUpdate:
        """Rank 0 only: re-read the edited deck and gate what changed.

        Raises :class:`DeckReloadError` on a deck that will not read or will
        not validate. ``check_for_deck_change`` contains that — the failure
        is reported and the solve continues — so it must not be caught here.
        """
        evicted = invalidate_cached_modules(self.dependencies)
        if evicted:
            logger.debug(
                "Reloading deck: dropped cached helper module(s) %s",
                ", ".join(sorted(evicted)),
            )

        from zutil.validate.zcfd_schema import inherited_settings

        report, dependencies = self._read()
        report.log_warnings()
        baseline = report.model.to_cpp_dict()
        inherited = inherited_settings(report.parameters)

        # Update in place: check_for_deck_change snapshots the mtimes after
        # this callback returns and broadcasts them, so adopting here is
        # what carries the new include chain (a renamed base file, say) to
        # every rank. Assigning a new object instead would leave the old
        # mtimes to be broadcast, and the same edit would re-report forever.
        self.dependencies.adopt(dependencies.mtimes)

        changed: dict = {}
        _compare_nested_structures(baseline, self.baseline, changed)

        # The baseline stays on rank 0 and is never broadcast. It holds the
        # deck's own callables — force-driver transforms, ramp functions —
        # which are defined inside the exec'd deck and so cannot be pickled
        # by qualified name. Only rank 0 ever diffs, so only rank 0 needs it.
        self.baseline = baseline
        old, self.inherited = self.inherited, inherited

        update = Parameters().partition_dynamic_update(changed)
        update.echoes = {
            key
            for key in changed
            if old is not None
            and old.source(key.split(":"))
            and inherited.source(key.split(":"))
        }
        return update.transferable()
