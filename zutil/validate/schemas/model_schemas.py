from pydantic import BaseModel, ConfigDict, model_validator
from typing import Optional, Dict, Union
from zutil.validate.schemas.base_types import (
    ConditionalOptional,
    When,
    zCFD_Field,
)
from zutil.validate.schemas.time_settings_schemas import APPLIES_WITH_PSEUDO_TIME

from . import boundary_conditions_schemas as bounds
from . import fluid_zones_schemas as fz
from . import output_schemas as out
from . import numerical_schemes_schemas as num
from . import block_versions as versions
from . import convergence_schemas as conv
from .initialisation_schemas import InitialisationModel
from .motion_schemas import ORDER
from .transform_schemas import TransformsModel


class ModelSettings(BaseModel):
    """
    Configuration for WHAT is being simulated on a specific mesh.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Core mesh identity
    mesh: str = zCFD_Field(
        title="Mesh",
        description=("Path to the HDF5 mesh file."),
    )

    # Mesh-specific configurations
    boundary_conditions: Dict[str, bounds.BoundaryCondition] = zCFD_Field(
        title="Boundary Conditions",
        description="Boundary conditions mapped to surface zones.",
    )
    fluid_zones: Dict[str, fz.Fluid_Zone_Base_Model] = zCFD_Field(
        default_factory=dict,
        title="Fluid Zones",
        description="Fluid zone models (e.g. porous, actuator disc).",
    )
    transforms: Optional[TransformsModel] = zCFD_Field(
        default=None,
        title="Transforms",
        description=f"Placement and motion of the mesh. {ORDER}",
    )

    # Mesh-specific overrides (if not provided, defaults from solver are used)
    initialisation: Optional[InitialisationModel] = zCFD_Field(
        default=None,
        title="Initialisation",
        description=(
            "The initialisation settings that differ on this mesh. Anything left out "
            "comes from the solver's initialisation."
        ),
    )
    # Same pseudo-time clause as the solver-level field: a per-mesh override
    # is just as inapplicable under GTS (time_settings is solver-only, so the
    # root-anchored path is correct here too).
    # The incompressible variant is the mesh-level one, which has no
    # `scheme`: the coupling scheme is chosen once for the whole run, so it
    # cannot be overridden per mesh.
    convergence_control: ConditionalOptional[  # type: ignore[valid-type]
        Union[
            conv.CompressibleConvergenceControl,
            conv.IncompressibleMeshConvergenceControl,
        ],
        When(APPLIES_WITH_PSEUDO_TIME),
    ] = zCFD_Field(
        default=None,
        title="Convergence Control",
        description=(
            "The convergence control settings that differ on this mesh. Anything left out "
            "comes from the solver's convergence control."
        ),
        variants=versions.MESH_CONVERGENCE_CONTROL,
    )
    numerical_scheme: Optional[num.NumericalScheme] = zCFD_Field(
        default=None,
        title="Numerical Scheme",
        description=(
            "The numerical scheme settings that differ on this mesh. Anything left out "
            "comes from the solver's numerical scheme."
        ),
        variants=versions.NUMERICAL_SCHEME,
    )
    output_settings: Optional[out.OutputModel] = zCFD_Field(
        default=None,
        title="Output Settings",
        description=(
            "The output settings that differ on this mesh. Anything left out "
            "comes from the solver's output settings."
        ),
    )

    @model_validator(mode="after")
    def check_duplicate_bc_zones(self):
        """Error if the same zone id is assigned to two different boundary conditions.

        Restored from the legacy voluptuous ``bc_zone_checker``
        (zcfd_validate_legacy/utils.py:95-126), which errored when a zone id
        appeared in more than one bc_* block.
        """
        zone_owner: Dict[int, str] = {}
        for bc_name, bc in self.boundary_conditions.items():
            for zone in getattr(bc, "zones", None) or []:
                owner = zone_owner.get(zone)
                if owner is not None and owner != bc_name:
                    raise ValueError(
                        f"Zone {zone} is assigned to both boundary conditions "
                        f"'{owner}' and '{bc_name}' — each zone may only "
                        "belong to one boundary condition."
                    )
                zone_owner[zone] = bc_name
        return self
