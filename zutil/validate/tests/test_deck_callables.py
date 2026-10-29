"""Every deck function is called with keywords and checked at validation.

Each callable gets two tests: a fragment written with the explicit
keyword-only signature validates, and a function returning the wrong shape
is rejected with a message naming what it returned and what it must return.
"""

import numpy as np
import pytest
from pydantic import ValidationError

import math

from zutil.validate.schemas.callables import (
    CFL_RAMP,
    FORCE_TRANSFORM,
    INITIAL_CONDITIONS,
    LEAF_AREA_DENSITY,
    MESH_DEFORMATION,
    MOMENTUM_SOURCE,
    is_vector3,
)
from zutil.validate.schemas.convergence_schemas import CFLModel
from zutil.validate.schemas.fluid_zones_schemas import (
    CanopyZone,
    IDWTransformZone,
    MomentumSourceZone,
    RBFTransformZone,
)
from zutil.validate.schemas.initialisation_schemas import InitialisationModel
from zutil.validate.schemas.output_schemas import ForceModel


class TestVector3:
    @pytest.mark.parametrize(
        "value",
        [
            (1.0, 2.0, 3.0),
            [1, 2, 3],
            np.array([1.0, 2.0, 3.0]),
            {"v1": 1.0, "v2": 2.0, "v3": 3.0},
        ],
    )
    def test_accepted(self, value):
        assert is_vector3(value)

    @pytest.mark.parametrize(
        "value",
        [
            {1.0, 2.0, 3.0},
            (1.0, 2.0),
            "xyz",
            {"x": 1.0, "y": 2.0, "z": 3.0},
            (1.0, None, 3.0),
            (True, 0.0, 0.0),
        ],
    )
    def test_rejected(self, value):
        assert not is_vector3(value)


class TestInitialConditions:
    def test_explicit_signature_validates(self):
        def shock(*, location, wall_distance, pressure, temperature, velocity):
            if location[0] < 0.5:
                return {"pressure": 2.0 * pressure, "velocity": [0.0, 0.0, 0.0]}
            return {"temperature": temperature}

        model = InitialisationModel(reference="IC_1", initial_conditions=shock)
        assert model.initial_conditions is shock

    def test_kwargs_signature_still_validates(self):
        def shock(**kwargs):
            return {"pressure": kwargs["pressure"]}

        InitialisationModel(reference="IC_1", initial_conditions=shock)

    def test_named_condition_still_validates(self):
        model = InitialisationModel(initial_conditions="IC_1")
        assert model.initial_conditions == "IC_1"

    def test_missing_keyword_rejected(self):
        def shock(*, location):
            return {}

        with pytest.raises(ValidationError) as excinfo:
            InitialisationModel(reference="IC_1", initial_conditions=shock)
        message = str(excinfo.value)
        assert "cannot take the keywords zCFD calls it with" in message
        assert (
            "def shock(*, location, wall_distance, pressure, temperature, velocity)"
            in message
        )

    def test_unknown_key_rejected(self):
        def shock(**kwargs):
            return {"density": 1.2}

        with pytest.raises(ValidationError) as excinfo:
            InitialisationModel(reference="IC_1", initial_conditions=shock)
        message = str(excinfo.value)
        assert "returned {'density': 1.2} (dict)" in message
        assert (
            "it must return a dict of any of 'pressure', 'temperature' and "
            "'velocity'; 'density' is not one of them." in message
        )

    def test_two_component_velocity_rejected(self):
        def shock(**kwargs):
            return {"velocity": (1.0, 0.0)}

        with pytest.raises(ValidationError) as excinfo:
            InitialisationModel(reference="IC_1", initial_conditions=shock)
        assert "; 'velocity' must be three numbers." in str(excinfo.value)

    def test_raising_function_rejected(self):
        def shock(**kwargs):
            return {"pressure": 1.0 / 0.0}

        with pytest.raises(ValidationError) as excinfo:
            InitialisationModel(reference="IC_1", initial_conditions=shock)
        message = str(excinfo.value)
        assert "raised ZeroDivisionError" in message
        assert "location=(0.0, 0.0, 0.0)" in message
        assert "and again when called with location=(0.3, 0.7, 0.2)" in message

    def test_inputs_are_what_the_solver_passes(self):
        assert list(INITIAL_CONDITIONS.inputs) == [
            "location",
            "wall_distance",
            "pressure",
            "temperature",
            "velocity",
        ]


class TestForceTransform:
    def test_explicit_signature_validates(self):
        def wind_axes(*, vector):
            x, y, z = vector
            return (x, -z, y)

        model = ForceModel(zones=[1], transform=wind_axes)
        assert model.transform is wind_axes

    def test_vector_dict_fallback_validates(self):
        def wind_axes(*, vector):
            return {"v1": vector[0], "v2": vector[1], "v3": vector[2]}

        ForceModel(zones=[1], transform=wind_axes)

    def test_positional_signature_rejected(self):
        def wind_axes(x, y, z):
            return (x, y, z)

        with pytest.raises(ValidationError) as excinfo:
            ForceModel(zones=[1], transform=wind_axes)
        message = str(excinfo.value)
        assert "The forces transform wind_axes(x, y, z)" in message
        assert "def wind_axes(*, vector)" in message

    def test_set_return_rejected(self):
        def wind_axes(*, vector):
            return {vector[0], vector[1], vector[2]}

        with pytest.raises(ValidationError) as excinfo:
            ForceModel(zones=[1], transform=wind_axes)
        message = str(excinfo.value)
        assert "(set)" in message
        assert "it must return the transformed vector as three numbers." in message


class TestMeshDeformation:
    @pytest.mark.parametrize("zone", [RBFTransformZone, IDWTransformZone])
    def test_explicit_signature_validates(self, zone):
        def pitch(*, location):
            x, y, z = location
            return [x, y, z + 0.01 * x]

        zone(zones=[1], func=pitch)

    @pytest.mark.parametrize("zone", [RBFTransformZone, IDWTransformZone])
    def test_scalar_return_rejected(self, zone):
        def pitch(*, location):
            return 0.0

        with pytest.raises(ValidationError) as excinfo:
            zone(zones=[1], func=pitch)
        message = str(excinfo.value)
        assert "The mesh deformation func pitch returned 0.0 (float)" in message


class TestMomentumSource:
    def test_explicit_signature_validates(self):
        def drive(*, locations):
            return [(0.0, 0.0, 0.08) for _ in locations]

        MomentumSourceZone(type="momentum", func=drive)

    def test_one_vector_for_all_rejected(self):
        def drive(*, locations):
            return (0.0, 0.0, 0.08)

        with pytest.raises(ValidationError) as excinfo:
            MomentumSourceZone(type="momentum", func=drive)
        assert (
            "it must return a list of one source vector of three numbers per "
            "location; it was given 2 locations." in str(excinfo.value)
        )


class TestLeafAreaDensity:
    def test_explicit_signature_validates(self):
        def lad(*, locations, wall_distances):
            return [5.75 if d < 20.0 else 0.0 for d in wall_distances]

        CanopyZone(function=lad)

    def test_one_tuples_rejected(self):
        def lad(*, locations, wall_distances):
            return [(5.75,) for _ in locations]

        with pytest.raises(ValidationError) as excinfo:
            CanopyZone(function=lad)
        assert (
            "it must return a list of one leaf area density per location; it "
            "was given 2 locations." in str(excinfo.value)
        )

    def test_old_single_list_signature_rejected(self):
        def lad(cell_centre_list):
            return [5.75 for _ in cell_centre_list]

        with pytest.raises(ValidationError) as excinfo:
            CanopyZone(function=lad)
        assert "def lad(*, locations, wall_distances)" in str(excinfo.value)


class TestCFLRamp:
    def test_explicit_signature_validates(self):
        def ramp(*, cycle, current_cfl):
            return min(1.0 * 1.005 ** (cycle - 1), 10.0)

        model = CFLModel(cfl=10.0, cfl_ramp=ramp)
        assert model.cfl_ramp is ramp

    def test_dict_return_validates(self):
        def ramp(*, cycle, current_cfl):
            return {"cfl": current_cfl, "cfl_turbulence": 0.5 * current_cfl}

        CFLModel(cfl=10.0, cfl_ramp=ramp)

    def test_positional_signature_rejected(self):
        def ramp(solve_cycle, current_cfl):
            return current_cfl

        with pytest.raises(ValidationError) as excinfo:
            CFLModel(cfl=10.0, cfl_ramp=ramp)
        assert "def ramp(*, cycle, current_cfl)" in str(excinfo.value)

    def test_unknown_key_rejected(self):
        def ramp(*, cycle, current_cfl):
            return {"cfl_mean": 1.0}

        with pytest.raises(ValidationError) as excinfo:
            CFLModel(cfl=10.0, cfl_ramp=ramp)
        assert "'cfl', 'cfl_turbulence' and 'cfl_coarse'" in str(excinfo.value)

    def test_solver_calls_ramp_with_keywords(self):
        CFL = pytest.importorskip("zcfd.utils.cfl").CFL

        def ramp(*, cycle, current_cfl):
            return 2.0 * cycle

        cfl = CFL(max_cfl=100.0)
        cfl.ramp_func = ramp
        cfl.update(3)
        assert cfl.current_cfl == 6.0

    def test_built_ramps_meet_contract(self):
        from zutil.validate.schemas.convergence_schemas import build_cfl_ramp

        built = build_cfl_ramp(
            [
                {"type": "exponential", "factor": 1.1, "initial": 1.0},
                {"type": "linear", "increment": 1.0, "start_cycle": 50},
            ]
        )
        CFL_RAMP.check(built)


class TestRemovedCallables:
    """Callables the solver never reached are no longer deck inputs."""

    def test_affine_transform_zone_rejected(self):
        from pydantic import TypeAdapter

        from zutil.validate.schemas.fluid_zones_schemas import Fluid_Zone_Base_Model

        with pytest.raises(ValidationError):
            TypeAdapter(Fluid_Zone_Base_Model).validate_python(
                {"type": "affine transform", "zones": [1]}
            )

    def test_mesh_transform_function_rejected(self):
        from zutil.validate.schemas.transform_schemas import TransformsModel

        with pytest.raises(ValidationError, match="transform_func"):
            TransformsModel(transform_func=lambda **kwargs: kwargs)

    def test_multigrid_ramp_is_a_cycle(self):
        from zutil.validate.schemas.numerical_schemes_schemas import MultigridConfig

        assert MultigridConfig(levels=3, multigrid_ramp=2.0).multigrid_ramp == 2
        with pytest.raises(ValidationError):
            MultigridConfig(levels=3, multigrid_ramp=lambda cycle: cycle)

    def test_modal_forcing_is_the_harmonic_initial_state(self):
        from zutil.validate.schemas.fsi_schemas import HarmonicInitialStateModel

        state = {
            "location": [0.0, 0.0, 0.0],
            "force": [0.0, 0.0, 1.0],
            "frequency": 5.0,
        }
        assert HarmonicInitialStateModel(**state).frequency == 5.0
        with pytest.raises(ValidationError):
            HarmonicInitialStateModel(**{**state, "force": lambda **kwargs: kwargs})


class TestSamplePoints:
    """A function singular on an axis is tried again off the axes before it is rejected."""

    def test_swirl_about_the_axis_validates(self):
        def swirl(*, location, wall_distance, pressure, temperature, velocity):
            x, y, _ = location
            r2 = x * x + y * y
            return {"velocity": (-y / r2, x / r2, 0.0)}

        InitialisationModel(reference="IC_1", initial_conditions=swirl)

    @pytest.mark.parametrize("zone", [RBFTransformZone, IDWTransformZone])
    def test_log_radius_deformation_validates(self, zone):
        def flare(*, location):
            x, y, z = location
            return (x, y, z + 0.01 * math.log(math.hypot(x, y)))

        zone(zones=[1], func=flare)

    def test_swirl_momentum_source_validates(self):
        def drive(*, locations):
            return [
                (-y / (x * x + y * y), x / (x * x + y * y), 0.0)
                for x, y, _ in locations
            ]

        MomentumSourceZone(type="momentum", func=drive)


class TestDescriptions:
    """Each field's help text is built from its contract, so it names the keywords validation binds."""

    @pytest.mark.parametrize(
        "model, name, contract",
        [
            (InitialisationModel, "initial_conditions", INITIAL_CONDITIONS),
            (ForceModel, "transform", FORCE_TRANSFORM),
            (RBFTransformZone, "func", MESH_DEFORMATION),
            (IDWTransformZone, "func", MESH_DEFORMATION),
            (MomentumSourceZone, "func", MOMENTUM_SOURCE),
            (CanopyZone, "function", LEAF_AREA_DENSITY),
            (CFLModel, "cfl_ramp", CFL_RAMP),
        ],
    )
    def test_description_names_the_call_and_the_return(self, model, name, contract):
        description = model.model_fields[name].description
        assert (
            f"called as {contract.call()}; it returns {contract.reads}" in description
        )
