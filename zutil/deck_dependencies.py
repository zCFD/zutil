"""Which deck files a config read actually executed.

v1 control decks were monolithic — one file, one mtime to poll. v2 decks
routinely split into a base plus overrides, chained through
``get_parameters_from_file`` (``testcases/testcases/dpw/dpw5_L2_sa_imp.py``
runs three deep), so watching the entry file alone silently misses an edit
to a base. Nothing in a deck declares that chain: it only becomes visible
by executing the deck.

``zutil.fileutils.load_config_namespace`` is the one place a deck file is
ever ``exec``'d, so it reports each file to whichever tracker is active.
Wrapping a read therefore yields the full set with no cooperation from the
deck and no user-facing declaration::

    with track_dependencies() as deps:
        params = get_parameters_from_file(control_file)

    # ... later, at report cadence ...
    if deps.changed():
        reload()

Tracking is opt-in: outside a :func:`track_dependencies` block the hooks
return immediately, so a normal solver startup pays nothing.

Detecting a change is not the same as being able to act on one. A deck
helper pulled in by plain ``import`` is cached in ``sys.modules``, so
re-executing the deck re-uses the stale module however many times the file
changes on disk; whoever re-reads the deck has to invalidate those modules
first. The set reported here is what changed, not what a re-read will
pick up.

Mtimes are stamped as each file is executed rather than when the block
closes, so an edit landing mid-read is still caught by the next poll.
A file that stops being stat-able — deleted, renamed, permissions changed
— counts as changed: a reload attempt that then fails reports loudly,
whereas treating it as unchanged would silently strand the run on stale
parameters. A file that could not be stat'd at read time in the first place
is recorded as absent and stays unchanged while it remains so, since there
is nothing to reload to; it registers if it later appears.

Not thread-safe by design: deck loading happens once per rank on the main
thread, and each MPI rank is its own process.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DeckChange",
    "DeckDependencies",
    "check_for_deck_change",
    "invalidate_cached_modules",
    "track_dependencies",
    "record_executed_file",
    "record_imported_siblings",
]


class DeckDependencies:
    """The deck files one config read executed, and their mtimes as read.

    Ordinarily built by :func:`track_dependencies` rather than directly;
    the explicit ``mtimes`` argument exists so a rank that received the set
    over MPI can reconstruct it (see the MPI-safe reload check).
    """

    def __init__(self, mtimes: Optional[dict] = None) -> None:
        self._mtimes: dict = dict(mtimes or {})

    @property
    def paths(self) -> frozenset:
        """Absolute, symlink-resolved paths of every file the read executed."""
        return frozenset(self._mtimes)

    @property
    def mtimes(self) -> dict:
        """Copy of the path -> mtime mapping, as stamped during the read."""
        return dict(self._mtimes)

    def record(self, path) -> None:
        """Note that ``path`` was executed, stamping its mtime now.

        First exec wins if a file is executed twice in one read: keeping
        the earlier mtime means an edit between the two execs still shows
        up as a change, where taking the later one would swallow it.
        """
        resolved = str(Path(path).resolve())
        if resolved in self._mtimes:
            return
        try:
            self._mtimes[resolved] = os.stat(resolved).st_mtime
        except OSError:
            # Unreadable at read time is odd but not ours to raise on —
            # record it as absent so the next poll reports it as changed.
            self._mtimes[resolved] = None

    def changed(self) -> List[str]:
        """Paths whose mtime no longer matches the one stamped at read time.

        Sorted, so log output and test assertions are stable. Empty means
        every file in the set is byte-for-byte the one that was read.
        """
        changed = []
        for path, was in self._mtimes.items():
            try:
                now = os.stat(path).st_mtime
            except OSError:
                now = None
            if now != was:
                changed.append(path)
        return sorted(changed)

    def restamp(self) -> None:
        """Re-read the mtimes of the files already in the set.

        Marks the current state on disk as the new baseline without
        re-running the deck. Used after a re-read that failed: without it
        the same broken deck reports as changed on every poll and floods
        the log, whereas re-baselining means the next report comes from the
        user's next edit — including the edit that fixes it.
        """
        for path in list(self._mtimes):
            try:
                self._mtimes[path] = os.stat(path).st_mtime
            except OSError:
                self._mtimes[path] = None

    def adopt(self, mtimes: dict) -> None:
        """Replace this set's contents with ``mtimes``.

        Lets a rank that did no filesystem work take on the state rank 0
        broadcast, so every rank's view stays identical.
        """
        self._mtimes = dict(mtimes)

    def __len__(self) -> int:
        return len(self._mtimes)

    def __contains__(self, path) -> bool:
        return str(Path(path).resolve()) in self._mtimes

    def __repr__(self) -> str:
        return f"DeckDependencies({len(self._mtimes)} files)"


# Trackers active right now, outermost first. A list rather than a single
# slot so that a nested track_dependencies() block feeds both itself and
# every enclosing tracker — an enclosing read did execute those files.
_active: List[DeckDependencies] = []


def record_executed_file(path) -> None:
    """Report a file as executed to every active tracker.

    Called by ``zutil.fileutils.load_config_namespace``. A no-op when no
    tracker is active, which is the normal solver startup path.
    """
    if not _active:
        return
    for tracker in _active:
        tracker.record(path)


def record_imported_siblings(directory) -> None:
    """Record already-imported modules living directly in ``directory``.

    A deck may pull in a helper sitting alongside it with a plain
    ``import`` — ``testcases/testcases/turbine/turbine_2p5D.py`` imports
    ``turbine_2p5D_zones``, and ``load_config_namespace`` puts the deck's
    directory on ``sys.path`` precisely so that works. Those helpers are as
    much a part of what the read executed as an explicit include, but
    Python's import machinery runs them rather than
    ``load_config_namespace``, so they have to be collected separately.

    Found by listing the directory and looking each candidate name up in
    ``sys.modules``, rather than by scanning ``sys.modules`` and resolving
    every module's path: a scan is O(modules imported) with a filesystem
    walk per module, which on a warm interpreter measured ~70 ms per deck
    read — some 98% of the cost of reading a deck at all, paid twice over
    for a two-file include chain. A directory listing plus a dict lookup
    per candidate is O(python files beside the deck) and costs nothing.

    Looked up rather than diffed across the read, because the second read
    of a deck re-uses the cached module and would import nothing at all.
    Only direct children of the directory count — that is exactly what
    putting it on ``sys.path`` exposes.
    """
    if not _active:
        return
    import sys

    root = Path(directory).resolve()
    try:
        entries = os.listdir(root)
    except OSError:
        return

    for entry in entries:
        if not entry.endswith(".py"):
            continue
        module = sys.modules.get(entry[:-3])
        if module is None:
            continue
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        # A module of the same name may have been imported from elsewhere;
        # only record it if it really is this file.
        try:
            if Path(filename).resolve() == root / entry:
                record_executed_file(filename)
        except OSError:
            continue


@contextmanager
def track_dependencies() -> Iterator[DeckDependencies]:
    """Collect the deck files executed inside this block.

    The yielded :class:`DeckDependencies` is populated as the read runs and
    is complete once the block exits.
    """
    deps = DeckDependencies()
    _active.append(deps)
    try:
        yield deps
    finally:
        _active.remove(deps)


class DeckChange:
    """The outcome of one reload poll, identical on every rank.

    ``changed`` lists the dependency files whose mtime moved (empty means
    no poll work happened). ``payload`` is whatever the ``on_change``
    callback returned, or ``None`` when nothing changed or the callback
    failed. ``failed`` distinguishes those two: a deck that changed but
    could not be re-read is not the same as a deck that did not change.
    """

    def __init__(
        self,
        changed: Optional[List[str]] = None,
        payload=None,
        failed: bool = False,
        error: Optional[str] = None,
        mtimes: Optional[dict] = None,
    ) -> None:
        self.changed = list(changed or [])
        self.payload = payload
        self.failed = failed
        self.error = error
        self.mtimes = dict(mtimes or {})

    def __repr__(self) -> str:
        if self.failed:
            return f"DeckChange(failed on {len(self.changed)} changed files)"
        return f"DeckChange({len(self.changed)} changed)"


def invalidate_cached_modules(dependencies: DeckDependencies) -> List[str]:
    """Drop any recorded dependency that is a cached module from ``sys.modules``.

    Deck files are ``exec``'d, so re-running one always reads it afresh —
    but a helper the deck pulled in with a plain ``import`` is cached, and
    re-executing the deck re-uses the stale module however many times the
    file changed on disk. Without this, an edit to ``turbine_2p5D_zones.py``
    is detected and then silently ignored.

    Returns the module names evicted, for logging. Only files already in
    the dependency set are touched, so nothing outside the deck's own
    directory can be evicted.
    """
    import sys

    targets = dependencies.paths
    # Cheap pre-filter: resolving every module's path is a filesystem walk
    # each, and almost none of them can match. Basenames are already in
    # hand, so only a genuine candidate pays for a resolve.
    basenames = {os.path.basename(path) for path in targets}

    evicted = []
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename or os.path.basename(filename) not in basenames:
            continue
        try:
            resolved = str(Path(filename).resolve())
        except OSError:
            continue
        if resolved in targets:
            del sys.modules[name]
            evicted.append(name)
    return evicted


def check_for_deck_change(dependencies: DeckDependencies, comm=None, on_change=None):
    """Poll the dependency set for edits, safely across MPI ranks.

    Only rank 0 touches the filesystem and the outcome is broadcast, so
    every rank takes the same branch on the same cycle. A per-rank check
    could diverge on NFS — ranks seeing the mtime flip at different moments,
    or one reading a half-written file — and deadlock the collective work
    that follows. This mirrors ``zcfd.utils.stop_file.check_for_stop_file``,
    which solves the same problem for the stop file.

    ``on_change`` runs on rank 0 only, and only when something changed; its
    return value is broadcast to every rank as ``payload``. It is where the
    re-read, re-validation, diff and whitelist gate belong — doing them on
    rank 0 alone means one filesystem read and one set of log messages
    rather than N of each.

    Exactly one broadcast happens per call whatever the outcome, including
    when ``on_change`` raises. That matters more than it looks: a collective
    is only safe if every rank reaches it the same number of times, so an
    exception escaping rank 0 here would leave the other ranks blocked in
    ``bcast`` forever. The callback is therefore contained — a deck saved
    half-written, or edited into a syntax error, is reported and the run
    carries on with the parameters it already has, rather than taking the
    solve down. Rank 0 then re-baselines the dependency set, so the failure
    is reported once per edit rather than on every poll.

    Whatever ``on_change`` returns must be picklable: ``comm.bcast`` sends it
    to the other ranks. Deck-derived data is a live hazard here — a control
    deck is ``exec``'d rather than imported, so any function it defines
    cannot be pickled by qualified name and would fail at this broadcast,
    a long way from the deck that caused it. Screen the payload before
    returning it rather than relying on it happening to be simple.

    Pass ``comm=None`` to skip the broadcast (serial use / unit tests).

    Side effect: ``dependencies`` is left holding the mtimes rank 0 saw, on
    every rank, so no rank's view can drift from the one that decided.
    """
    rank_zero = comm is None or comm.Get_rank() == 0

    result = None
    if rank_zero:
        changed = dependencies.changed()
        payload = None
        failed = False
        error = None
        if changed and on_change is not None:
            try:
                payload = on_change()
            except Exception as exc:  # noqa: BLE001 — see docstring
                failed = True
                error = f"{type(exc).__name__}: {exc}"
                logger.debug("Deck changed but could not be re-read: %s", error)
                # Re-baseline so the same broken deck is not re-reported
                # every poll; the user's next edit reports afresh.
                dependencies.restamp()
        result = DeckChange(
            changed=changed,
            payload=payload,
            failed=failed,
            error=error,
            mtimes=dependencies.mtimes,
        )

    if comm is not None:
        result = comm.bcast(result, root=0)
        dependencies.adopt(result.mtimes)

    return result
