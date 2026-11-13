import json

import pytest

from zutil.docs.fragments import check_pages


def entry(path, kind=None, options=None, fields=None):
    return {
        "path": path,
        "facts": {"type": kind},
        "options": [{"value": o} for o in options] if options else None,
        "fields": fields or [],
    }


PAGES = {
    "solver/convergence-control": {
        "kind": "node",
        "deck_path": ["solver", "convergence_control"],
        "entries": [
            entry(["cycles"], "int"),
            entry(["cfl"], "float"),
            entry(
                ["scheme"],
                "settings block",
                fields=[
                    entry(
                        ["scheme", "name"], "choice", ["runge kutta", "implicit euler"]
                    )
                ],
            ),
        ],
    },
    "solver/convergence-control/ramp": {
        "kind": "node",
        "deck_path": ["solver", "convergence_control", "ramp", "<n>"],
        "entries": [
            entry(["initial"], "float"),
            entry(["type"], "choice", ["linear", "geometric"]),
        ],
    },
    "solver/time-settings": {
        "kind": "node",
        "deck_path": ["solver", "time_settings"],
        "entries": [entry(["type"], "choice", ["steady", "unsteady"])],
    },
    "solver/solver-settings": {
        "kind": "node",
        "deck_path": ["solver", "solver_settings"],
        "entries": [
            entry(
                ["linear_solver_options"],
                "settings block",
                fields=[
                    entry(["linear_solver_options", "flow"], "settings block"),
                ],
            )
        ],
    },
    "solver/output-settings": {
        "kind": "node",
        "deck_path": ["solver", "output_settings"],
        "entries": [
            entry(
                ["report"],
                "settings block",
                fields=[
                    entry(
                        ["report", "monitor"],
                        "named settings blocks",
                        fields=[
                            entry(
                                ["report", "monitor", "point"],
                                "vector (3 floats)",
                            ),
                        ],
                    ),
                ],
            )
        ],
    },
    "solver/reference-conditions": {
        "kind": "node",
        "deck_path": ["solver", "reference_conditions", "<name>"],
        "entries": [entry(["temperature"], "float"), entry(["pressure"], "float")],
    },
    "model": {
        "kind": "node",
        "deck_path": ["model", "<mesh>"],
        "entries": [
            entry(["mesh"], "text"),
            entry(["convergence_control"], "one of 2 settings blocks"),
        ],
    },
    "model/boundary-conditions/wall": {
        "kind": "type",
        "deck_path": ["model", "<mesh>", "boundary_conditions", "<name>"],
        "entries": [
            entry(["type"], "choice", ["wall"]),
            entry(["zones"], "list of zone ids"),
            entry(["kind"], "choice", ["no slip", "slip", "wall function"]),
        ],
    },
    "model/boundary-conditions/inflow": {
        "kind": "type",
        "deck_path": ["model", "<mesh>", "boundary_conditions", "<name>"],
        "entries": [
            entry(["type"], "choice", ["inflow"]),
            entry(["zones"], "list of zone ids"),
            entry(["reference"], "str"),
        ],
    },
}


@pytest.fixture
def reference(tmp_path):
    for route, page in PAGES.items():
        path = tmp_path / "ref" / "reference" / f"{route}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(page, route=f"reference/{route}")))
    return tmp_path / "ref"


def problems(tmp_path, reference, body, lang="python"):
    pages = tmp_path / "pages"
    pages.mkdir(exist_ok=True)
    (pages / "page.md").write_text(f"Intro.\n\n```{lang}\n{body}\n```\n")
    return [str(p) for p in check_pages(pages, reference)]


def test_a_fragment_that_matches_passes(tmp_path, reference):
    assert (
        problems(
            tmp_path,
            reference,
            '{"convergence_control": {"cycles": 50, "cfl": 2.0}}',
            "json",
        )
        == []
    )


def test_a_renamed_key_fails_with_its_page_line_and_path(tmp_path, reference):
    assert problems(
        tmp_path,
        reference,
        '{"convergence_control": {"cycles": 50, "cfll": 2.0}}',
        "json",
    ) == ["page.md:3: 'solver › convergence_control › cfll' is not in the reference"]


def test_an_option_the_reference_does_not_list_fails(tmp_path, reference):
    assert problems(tmp_path, reference, '{"time_settings": {"type": "steddy"}}') == [
        "page.md:3: 'solver › time_settings › type' is 'steddy'; the reference lists 'steady', 'unsteady'"
    ]


def test_deck_spelling_is_accepted(tmp_path, reference):
    assert (
        problems(
            tmp_path,
            reference,
            '"convergence control": {"scheme": {"name": "implicit euler"}},',
        )
        == []
    )


def test_elisions_variables_and_calls_are_not_checked(tmp_path, reference):
    body = """"convergence_control": {
    ...
    "cycles": n_cycles,
    "scheme": {...},
    "cfl": math.sqrt(4.0),
    ..
},"""
    assert problems(tmp_path, reference, body) == []


def test_instance_names_are_free_and_their_type_selects_the_keys(tmp_path, reference):
    ok = '{"BC_1": {"type": "wall", "zones": [1], "kind": "no slip"}, "BC_2": {"type": "inflow", "reference": "IC_1"}}'
    assert problems(tmp_path, reference, ok) == []
    wrong = '{"BC_2": {"type": "inflow", "kind": "no slip"}}'
    assert problems(tmp_path, reference, wrong) == [
        "page.md:3: 'model › boundary_conditions › BC_2 › kind' is not a setting of a 'inflow' block"
    ]


def test_a_named_map_inside_a_block_takes_any_instance_name(tmp_path, reference):
    body = '{"report": {"monitor": {"probe_1": {"point": [0.0, 0.0, 0.0]}}}}'
    assert problems(tmp_path, reference, body) == []


def test_a_mesh_block_takes_the_solver_block_keys(tmp_path, reference):
    ok = '{"model": {"rotor": {"mesh": "rotor.h5", "convergence_control": {"cycles": 50}}}}'
    assert problems(tmp_path, reference, ok) == []
    wrong = '{"model": {"rotor": {"mesh": "rotor.h5", "convergence_control": {"cycle": 50}}}}'
    assert problems(tmp_path, reference, wrong) == [
        "page.md:3: 'model › rotor › convergence_control › cycle' is not in the reference"
    ]


def test_a_free_settings_block_takes_any_key(tmp_path, reference):
    body = '{"linear_solver_options": {"flow": {"-ksp_type": "fgmres"}}}'
    assert problems(tmp_path, reference, body) == []


def test_an_assignment_to_parameters_sets_the_root(tmp_path, reference):
    body = 'parameters["solver"]["reference_conditions"] = {"IC_1": {"temperature": 300.0, "pressur": 1.0}}'
    assert problems(tmp_path, reference, body) == [
        "page.md:3: 'solver › reference_conditions › IC_1 › pressur' is not in the reference"
    ]


def test_a_fence_naming_a_file_is_not_a_fragment(tmp_path, reference):
    assert (
        problems(
            tmp_path,
            reference,
            '{"config_version": 2, "solver": {}}',
            "json file=amgx.json",
        )
        == []
    )


def test_a_fence_that_cannot_be_read_is_reported(tmp_path, reference):
    assert problems(tmp_path, reference, '"cycles": 5 "cfl": 2') == [
        "page.md:3: the fragment cannot be read as a deck dict"
    ]


def test_code_without_a_parameters_assignment_is_not_a_fragment(tmp_path, reference):
    assert problems(tmp_path, reference, "def f(x):\n    return x * 2") == []


def test_a_list_page_checks_each_item(tmp_path, reference):
    ok = '{"convergence_control": {"ramp": [{"initial": 1.0, "type": "linear"}]}}'
    assert problems(tmp_path, reference, ok) == []
    wrong = '{"convergence_control": {"ramp": [{"type": "lineer"}]}}'
    assert problems(tmp_path, reference, wrong) == [
        "page.md:3: 'solver › convergence_control › ramp › type' is 'lineer'; the reference lists 'linear', 'geometric'"
    ]
