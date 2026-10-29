"""
Characterisation test for the buoyancy-input schema placement.

``Reference::set`` reads ``gravity``, ``latitude``, ``planet_rotation`` and
``thermal_expansion_coeff`` from ``parameters.fluidProperties()`` -- i.e. the
``solver.fluid_properties`` sub-dict of the dict produced by
``ZCFDInputModelBase.to_cpp_dict()``. All four fields live on
``FluidModelBase`` (zutil/zutil/validate/schemas/fluid_model_schemas.py).

Validating them onto ``ReferenceConditionsBase`` instead -- a dict the solver
never reads for these -- would let a buoyancy_cavity* case run without
buoyancy while still reporting a green validation pass. This test asserts the
values reach the dict the C++ actually reads.
"""

from zutil.validate.zcfd_schema import ZCFDInputModelBase


def _incompressible_buoyancy_config():
    """Minimal incompressible viscous config carrying gravity + beta.

    thermal_expansion_coeff requires solver.equations.solve_energy=True on
    the incompressible solver (FluidModelBase.thermal_expansion_coeff's
    applies_when clause) -- set explicitly here, faithful to how the
    buoyancy_cavity* control files map their v1 "solve energy": True.
    """
    return {
        "config_version": 2,
        "solver": {
            "solver_settings": {"type": "incompressible"},
            "equations": {"type": "viscous", "solve_energy": True},
            "numerical_scheme": {"order": "second"},
            "fluid_properties": {
                "material": "air",
                "gravity": [0.0, -9.81, 0.0],
                "thermal_expansion_coeff": 1.0 / 300.0,
                "latitude": 45.0,
                "planet_rotation": 7.292115e-5,
            },
            "reference_conditions": {
                "IC_1": {
                    "temperature": 300.0,
                    "pressure": 101325.0,
                    "v": [1.0, 0.0, 0.0],
                    "reynolds_no": 100.0,
                },
            },
            "initialisation": {"initial_conditions": "IC_1"},
            "time_settings": {"type": "steady"},
            "convergence_control": {
                "cycles": 10,
                "scheme": {"name": "coupled"},
            },
            "output_settings": {
                "solution": {"frequency": 10},
                "report": {"frequency": 10},
            },
        },
        "model": {
            "base": {
                "mesh": "base.h5",
                "boundary_conditions": {
                    "bc1": {"type": "wall", "kind": "no slip", "zones": [1]},
                },
            }
        },
    }


class TestBuoyancyInputsReachFluidPropertiesDict:
    def test_incompressible_gravity_and_beta_land_in_fluid_properties(self):
        model = ZCFDInputModelBase.model_validate(_incompressible_buoyancy_config())
        cpp_dict = model.to_cpp_dict()

        fluid_props = cpp_dict["solver"]["fluid_properties"]
        assert fluid_props["gravity"] == [0.0, -9.81, 0.0]
        assert fluid_props["thermal_expansion_coeff"] == 1.0 / 300.0
        assert fluid_props["latitude"] == 45.0
        assert fluid_props["planet_rotation"] == 7.292115e-5

        # And not on reference_conditions, which Reference::set never reads
        # for these -- the historical silent-drop bug.
        ic_1 = cpp_dict["solver"]["reference_conditions"]["ic_1"]
        assert "gravity" not in ic_1
        assert "thermal_expansion_coeff" not in ic_1
