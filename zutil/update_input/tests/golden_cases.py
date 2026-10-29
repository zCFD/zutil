"""The golden v1 decks: where each lives, its mesh, and the patterns it covers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

GOLDEN_DIR = Path(__file__).parent / "decks" / "golden"


@dataclass(frozen=True)
class GoldenCase:
    name: str
    deck: str
    mesh: Optional[str]
    develop: Optional[str]
    patterns: tuple[str, ...]

    @property
    def deck_path(self) -> Path:
        return GOLDEN_DIR / self.deck


GOLDEN_CASES: list[GoldenCase] = [
    GoldenCase(
        name="rotating_cylinder_refonly",
        deck="rotating_cylinder_refonly/cylinder_rotating.py",
        mesh="CYLINDER/cylinder.h5",
        develop="testcases/testcases/euler_rotating_cylinder/cylinder_rotating.py",
        patterns=(
            "rotating fluid zone on a still mesh, written as a rotating fluid zone on its cell zones",
            "BC by ref with no zone",
            "mesh transform matrix from a helper",
            "cfl ramp factor",
            "monitor point",
            "write output frequency dict",
        ),
    ),
    GoldenCase(
        name="plate_derived_ic",
        deck="plate_derived_ic/plate_coarse.py",
        mesh="PLATE/plate_coarse.h5",
        develop="testcases/testcases/plate/plate_coarse.py",
        patterns=(
            "derived IC with total pressure and temperature ratios feeding an inflow",
            "steady (total time == time step)",
            "zutil helper call",
            "RANS sst",
        ),
    ),
    GoldenCase(
        name="plate_wallfunction",
        deck="plate_wallfunction/plate_coarse_visc_wf.py",
        mesh="PLATE/plate_coarse.h5",
        develop="testcases/testcases/plate/plate_coarse_visc_wf.py",
        patterns=(
            "wall function",
            "derived IC left unreferenced",
            "steady (total time == time step)",
            "viscous equations",
        ),
    ),
    GoldenCase(
        name="shocktube_gts",
        deck="shocktube_gts/shocktube_fv_first.py",
        mesh="SHOCKTUBE/ShockTube.h5",
        develop="testcases/testcases/shocktube/shocktube_fv_first.py",
        patterns=(
            "global timestepping",
            "initial block with func callable",
            "runge kutta scheme",
            "entropy fix",
        ),
    ),
    GoldenCase(
        name="shocktube_dts",
        deck="shocktube_dts/shocktube_fv_second_imp.py",
        mesh="SHOCKTUBE/ShockTube.h5",
        develop="testcases/testcases/shocktube/shocktube_fv_second_imp.py",
        patterns=(
            "dual time stepping",
            "initial block with func callable",
            "implicit euler scheme",
        ),
    ),
    GoldenCase(
        name="shocktube_tf_variant",
        deck="shocktube_tf_variant/shocktube_fv_second_ets_tf.py",
        mesh="SHOCKTUBE/ShockTube.h5",
        develop=None,
        patterns=(
            "base plus variant through get_parameters_from_file with namespace",
            "global timestepping",
            "force transform callable",
            "initial block with func callable",
            "variant overrides keys of the base",
        ),
    ),
    GoldenCase(
        name="bolund_abl",
        deck="bolund_abl/bolund.py",
        mesh="BOLUND/bolund.h5",
        develop="testcases/testcases/bolund/bolund.py",
        patterns=(
            "ABL profile",
            "wall function with roughness field",
            "zutil helper call",
            "viscosity given directly",
            "turbulence model parameter",
        ),
    ),
    GoldenCase(
        name="bump_namespace",
        deck="bump_namespace/3DBump_5.py",
        mesh="3DBUMP/grid5x45x21_vol.h5",
        develop="testcases/testcases/3DBump/3DBump_5.py",
        patterns=(
            "get_parameters_from_file with namespace",
            "ramp func callable",
            "massflow inflow BC",
            "derived ICs with mass flow rate and ratios",
            "report massflow",
            "zutil helper call",
        ),
    ),
    GoldenCase(
        name="bump_incomp_namespace",
        deck="bump_incomp_namespace/3DBump_incomp_5.py",
        mesh="3DBUMP/grid5x45x21_vol.h5",
        develop="testcases/testcases/3DBump/3DBump_incomp_5.py",
        patterns=(
            "incompressible",
            "get_parameters_from_file with namespace",
            "zutil helper call",
        ),
    ),
    GoldenCase(
        name="naca_force_transform",
        deck="naca_force_transform/n0012_257_a15p0.py",
        mesh="NACA0012/naca0012_0257.h5",
        develop=None,
        patterns=(
            "force transform callable",
            "steady (total time == time step)",
            "zutil helper calls",
            "cfl ramp factor",
            "multigrid",
        ),
    ),
    GoldenCase(
        name="naca_ic_callable",
        deck="naca_ic_callable/n0012_257_cL_target_1_imp.py",
        mesh="NACA0012/naca0012_0257.h5",
        develop="testcases/testcases/naca0012/n0012_449_cL_target_1_imp.py",
        patterns=(
            "IC given as a callable, a DECIDE naming the farfield target",
            "driven parameter",
            "module level state and classes",
        ),
    ),
    GoldenCase(
        name="pipe_massflow",
        deck="pipe_massflow/pipe_massflow.py",
        mesh="PIPE/pipe.h5",
        develop="testcases/testcases/pipe/pipe_massflow.py",
        patterns=(
            "massflow inflow BC",
            "derived ICs with ratios",
            "lower case IC and BC keys",
            "local timestepping",
            "report massflow",
            "zutil helper call inside a function",
        ),
    ),
    GoldenCase(
        name="powered_restart",
        deck="powered_restart/wingbody_powered.py",
        mesh=None,
        develop=None,
        patterns=(
            "scale",
            "restart from another casename, ignoring its history",
            "steady (total time == time step)",
            "cfl ramp factor",
            "derived ICs feeding massflow outflows, a massflow inflow and two "
            "default inflows, with ratios and mass flow rates written as expressions",
            "BCs with empty zone lists",
            "positional (x, y, z) force transform shared by several force reports",
            "named massflow reports",
            "RANS sst with qcr",
            "expected v2 deck and report checked in under expected/",
        ),
    ),
    GoldenCase(
        name="cylinder_incomp_refonly",
        deck="cylinder_incomp_refonly/cylinder_incomp.py",
        mesh="CYLINDER/cylinder_mod.h5",
        develop="testcases/testcases/cylinder/cylinder_incomp.py",
        patterns=(
            "incompressible",
            "BC by ref with no zone",
            "linear solver options",
            "no scheme block",
        ),
    ),
    GoldenCase(
        name="overset_chain_a1",
        deck="overset_chain_a1/cylinderA1.py",
        mesh="OVERSET_ROTATIONS/cylinder_origin.h5",
        develop=None,
        patterns=(
            "three-level base chain through get_parameters_from_file with namespace",
            "rotating fluid zone on a still mesh, written as a rotating fluid zone on its cell zones",
            "mesh transform matrix from a callable",
            "overset BC",
            "zutil helper calls",
        ),
    ),
    GoldenCase(
        name="euler_rotating_cylinder_group",
        deck="euler_rotating_cylinder_group/overset.py",
        mesh=None,
        develop="testcases/testcases/euler_rotating_cylinder/euler_rotating_cylinder.py",
        patterns=(
            "overset group of two decks",
            "models that disagree on a solver-only setting",
            "output settings that differ written whole in one model",
            "rotating fluid zone with moving mesh, written as transforms > motion",
        ),
    ),
    GoldenCase(
        name="traintunnel_group",
        deck="traintunnel_group/override_roe.py",
        mesh=None,
        develop="testcases/testcases/traintunnel/traintunnel.py",
        patterns=(
            "overset group of two decks",
            "translating fluid zone with a translation function, a ramp time decision",
            "sponge layer on the farfield of one model",
            "mesh transform matrix from a helper",
        ),
    ),
] + [
    GoldenCase(
        name=f"overset_rotations_case{case}",
        deck=f"overset_rotations_groups/case{case}.py",
        mesh=None,
        develop=f"testcases/testcases/overset_rotations/case{case}.py",
        patterns=(
            "overset group of five decks sharing one base chain",
            "overset layers and sponge layer on the models that have the boundary",
            "one mesh shared by several models",
        ),
    )
    for case in "ABCD"
]
