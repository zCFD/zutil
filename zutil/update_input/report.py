"""The conversion report (markdown) and the console summary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .model import Conversion, Decision, Disposition, V1Path

SEP = " > "

_DROPPED_SHOWN = 3


@dataclass(frozen=True)
class Names:
    """The file names the report and the summary talk about."""

    deck: str
    output: str
    report: str
    config_version: Any = 0
    bases: tuple[tuple[str, str], ...] = ()
    unvalidated: str = ""


def join(path: Iterable[Any]) -> str:
    return SEP.join(str(k) for k in path)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _counts(c: Conversion) -> dict[str, int]:
    counts = {"moved": 0, "respelled": 0, "dropped": 0, "decide": 0}
    for d in c.dispositions:
        counts[d.kind] += 1
    counts["decide"] = len(c.decisions)
    return counts


def v1_name(c: Conversion, v1_path: V1Path) -> str:
    """A v1 key path as the deck spelt it."""
    leaf = c.v1.leaf(*v1_path)
    return join(leaf.keys if leaf is not None else v1_path)


def _value_text(c: Conversion, v1_path: V1Path) -> str:
    leaf = c.v1.leaf(*v1_path)
    if leaf is None:
        return ""
    return leaf.source if leaf.source is not None else repr(leaf.value)


def _options(decision: Decision) -> str:
    return " or ".join(repr(o) for o in decision.options)


def _status(c: Conversion, names: Names, validated: bool) -> str:
    if c.decisions:
        open_ = _plural(len(c.decisions), "decision")
        return f"draft: {open_} open; replace each DECIDE string"
    if validated:
        return "valid config version 2"
    return names.unvalidated or "not validated"


def summary(c: Conversion, names: Names, validated: bool) -> str:
    """The short account printed to the console after a conversion."""
    counts = _counts(c)
    lines = [
        f"Converting {names.deck} (config_version {names.config_version}) to config_version 2",
        "",
    ]
    if counts["moved"]:
        lines.append(
            f"  moved      {counts['moved']:>3} {'key' if counts['moved'] == 1 else 'keys'}"
        )
    if counts["respelled"]:
        n = counts["respelled"]
        lines.append(f"  re-spelled {n:>3} {'value' if n == 1 else 'values'}")
    if counts["dropped"]:
        n = counts["dropped"]
        dropped = [d for d in c.dispositions if d.kind == "dropped"]
        shown = [
            f"{v1_name(c, d.v1_path)}: {d.reason}" for d in dropped[:_DROPPED_SHOWN]
        ]
        if len(dropped) > _DROPPED_SHOWN:
            shown.append("...")
        lines.append(
            f"  dropped    {n:>3} {'key' if n == 1 else 'keys'}   ({' · '.join(shown)})"
        )
    if counts["decide"]:
        n = counts["decide"]
        lines.append(f"  DECIDE     {n:>3} {'key' if n == 1 else 'keys'}")
    if c.notes:
        lines.append(f"  notes      {len(c.notes):>3}   (in the report)")

    if c.decisions:
        lines.append("")
        for decision in c.decisions:
            lines.extend(_summary_decision(c, decision))

    lines.append(f"Wrote {names.output} ({_status(c, names, validated)})")
    lines.append(f"Wrote {names.report} (every key and where it went)")
    return "\n".join(lines)


def _summary_decision(c: Conversion, decision: Decision) -> list[str]:
    indent = " " * (len(str(decision.id)) + 4)
    if decision.v1_paths:
        heads = [f"{v1_name(c, p)} = {_value_text(c, p)}" for p in decision.v1_paths]
    else:
        heads = [join(decision.v2_path)]
    lines = [f"  {decision.id}. {heads[0]}"]
    lines.extend(indent + h for h in heads[1:])
    question = decision.question
    if decision.options:
        question += f" ({_options(decision)})"
    lines.append(indent + question)
    lines.append("")
    return lines


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _what_happened(d: Disposition) -> str:
    if d.kind == "dropped":
        return f"dropped: {d.reason}"
    if d.kind == "decide":
        return f"DECIDE, item {d.decision_id}"
    word = "re-spelled" if d.kind == "respelled" else "moved"
    return f"{word} ({d.reason})" if d.reason else word


def _where(d: Disposition) -> str:
    return f"`{join(d.v2_path)}`" if d.v2_path else ""


def markdown(c: Conversion, names: Names, validated: bool) -> str:
    """The full report: every v1 key, what happened to it, and every open decision."""
    counts = _counts(c)
    out = [f"# {names.deck} to config version 2", ""]
    out.append(
        f"update_input converted `{names.deck}` (config_version {names.config_version}) "
        f"and wrote `{names.output}`."
    )
    if c.decisions:
        out.append(
            f"The new deck is a draft: {_plural(len(c.decisions), 'decision')} below must be "
            "made before it validates or runs."
        )
    elif validated:
        out.append("The new deck passed config version 2 validation.")
    elif names.unvalidated:
        out.append(f"The new deck was {names.unvalidated}.")
    out.append("")

    out += ["## Files written", ""]
    for output, report_name in names.bases:
        out.append(
            f"- `{output}` and `{report_name}`, the base this deck takes its parameters from"
        )
    out.append(f"- `{names.output}` and `{names.report}`, this deck and this report")
    out.append("")

    out += ["## Counts", "", "| What happened | Count |", "|---|---|"]
    out.append(f"| moved | {counts['moved']} |")
    out.append(f"| re-spelled | {counts['respelled']} |")
    out.append(f"| dropped | {counts['dropped']} |")
    out.append(f"| DECIDE | {counts['decide']} |")
    out.append("")

    by_path: dict[tuple, Disposition] = {d.v1_path: d for d in c.dispositions}
    out += [
        "## Every v1 key",
        "",
        "| v1 key | What happened | Where it went |",
        "|---|---|---|",
    ]
    for leaf in c.v1:
        d: Optional[Disposition] = by_path.get(leaf.path)
        name = f"`{_cell(join(leaf.keys))}`"
        if d is None:
            out.append(f"| {name} | not converted |  |")
        else:
            out.append(f"| {name} | {_cell(_what_happened(d))} | {_cell(_where(d))} |")
    out.append("")

    if c.decisions:
        out += ["## Decisions", ""]
        for decision in c.decisions:
            out.append(f"### {decision.id}. {decision.question}")
            out.append("")
            out.append(f"v2 key: `{join(decision.v2_path)}`")
            out.append("")
            if decision.options:
                out.append(f"Options: {_options(decision)}")
                out.append("")
            if decision.details:
                out.append(decision.details)
                out.append("")
            if decision.v1_paths:
                keys = ", ".join(
                    f"`{v1_name(c, p)}` = `{_value_text(c, p)}`"
                    for p in decision.v1_paths
                )
                out.append(f"v1 keys: {keys}")
                out.append("")

    if c.notes:
        out += ["## Notes", ""]
        for note in c.notes:
            keys = ", ".join(f"`{v1_name(c, p)}`" for p in note.v1_paths)
            where = f" ({keys})" if keys else ""
            out.append(f"- {note.text}{where}")
        out.append("")

    return "\n".join(out)
