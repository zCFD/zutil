from pydantic import BaseModel, ConfigDict, Field, Tag
from typing import Literal, List, Optional, Union, Annotated
from zutil.validate.schemas.base_types import (
    ConditionalChoice,
    NotIn,
    When,
    zCFD_Field,
)
from .base_types import Vector3d
from .transform_schemas import RBFTransformConfig, IDWTransformConfig


_FSI_TYPE = (
    "The structural coupling: `modal_model` uses the built-in Nastran modal "
    "model, `generic_fsi` couples to an external structural solver."
)


class HarmonicInitialStateModel(BaseModel):
    """Start each mode on its harmonic response to a point force.

    The solver reads the block once, at the start of a run that is not a
    restart. It applies no force while the run marches.
    """

    model_config = ConfigDict(extra="forbid")

    location: Vector3d = zCFD_Field(
        ...,
        title="Location",
        description=(
            "Point the force acts at; the force there is written to the modal history"
        ),
    )
    force: Vector3d = zCFD_Field(
        ...,
        title="Force",
        description=(
            "Force vector (N); its magnitude sets each mode's initial "
            "displacement and velocity"
        ),
    )
    frequency: float = zCFD_Field(
        ..., title="Frequency", description="Frequency of the harmonic force (Hz)"
    )


class ModalModelBase(BaseModel):
    """Base configuration for Modal Model."""

    model_config = ConfigDict(extra="forbid")

    file_format: Literal["nastran"] = zCFD_Field(
        ..., title="File Format", description="Structural file format"
    )
    nastran_casename: str = zCFD_Field(
        ...,
        title="Nastran Casename",
        description="Nastran case name (without extension)",
    )
    scale: Optional[List[float]] = zCFD_Field(
        None, title="Scale", description="Scaling factors [x, y, z]"
    )
    mode_mapping_max_distance: float = zCFD_Field(
        ...,
        title="Mode Mapping Max Distance",
        description="Max distance for mode mapping",
    )
    # How the structural mode shapes are interpolated onto the fluid surface.
    # Distinct from the mesh-deformation power/n_nearest inherited by the RBF
    # and IDW subclasses: those morph the volume mesh, these map the modes.
    mode_mapping_power: float = zCFD_Field(
        default=1.0,
        gt=0,
        title="Mode Mapping Power",
        description="Inverse-distance power for mapping mode shapes to the surface",
        advanced=True,
    )
    mode_mapping_pure_idw: bool = zCFD_Field(
        default=True,
        title="Mode Mapping Pure IDW",
        description="Use pure inverse-distance weighting for the mode mapping",
        advanced=True,
    )
    mode_mapping_n_nearest: int = zCFD_Field(
        default=4,
        ge=1,
        title="Mode Mapping N Nearest",
        description="Number of nearest structural points used per surface point",
        advanced=True,
    )
    # Read unguarded by the modal FSI path, so it must always be
    # present; 0.0 disables recalculation.
    recalculate_rbfs_alpha_fraction: float = zCFD_Field(
        default=0.0,
        ge=0.0,
        title="Recalculate RBFs Alpha Fraction",
        description=(
            "Deformation fraction of the RBF support radius at which the "
            "base-point set is rebuilt; 0 never rebuilds it"
        ),
        advanced=True,
    )
    # Read unguarded by the modal FSI morphing setup, so they
    # must always be present.
    support_radius: float = zCFD_Field(
        default=1.0,
        gt=0,
        title="Support Radius",
        description=(
            "RBF support radius, also the mode-shape deformation scale "
            "used when normalising mode shapes for morphing"
        ),
    )
    newmark_alpha: float = zCFD_Field(
        default=0.25,
        title="Newmark Alpha",
        description="Newmark integrator alpha parameter",
        advanced=True,
    )
    newmark_delta: float = zCFD_Field(
        default=0.5,
        title="Newmark Delta",
        description="Newmark integrator delta parameter",
        advanced=True,
    )
    update_frequency: int = zCFD_Field(
        default=1,
        ge=1,
        title="Update Frequency",
        description="Cycles between structural updates: the modes are marched and the mesh deformed once every this many solver cycles. With several modal models the smallest value applies to all of them.",
        advanced=True,
    )
    fluid_force_scaling: float = zCFD_Field(
        ..., title="Fluid Force Scaling", description="Scaling factor for fluid forces"
    )
    mode_list: List[int] = zCFD_Field(
        ..., title="Mode List", description="List of modes to include"
    )
    modal_damping: List[float] = zCFD_Field(
        ..., title="Modal Damping", description="Damping factors for each mode"
    )
    harmonic_initial_state: Optional[HarmonicInitialStateModel] = zCFD_Field(
        None,
        title="Harmonic Initial State",
        description=(
            "Starts every mode on its harmonic response to a point force, "
            "instead of at rest"
        ),
    )

    # Type discriminator


ModalModelType = ConditionalChoice[
    Literal["modal_model"],
    {
        "modal_model": (
            When({"<parent>.zones": NotIn([])}),
            "a modal model moves the zones of its boundary condition, so list at least one under 'zones'",
        )
    },
]


class RBFModalModel(ModalModelBase, RBFTransformConfig):
    """Radial Basis Function (RBF) Multi-scale transformation."""

    type: ModalModelType = zCFD_Field("modal_model", description=_FSI_TYPE)
    transform_type: Literal["rbf_multiscale"] = zCFD_Field(
        "rbf_multiscale", title="Transform Type"
    )

    # Enforce mandatory fields for FSI RBF
    base_point_fraction: float = zCFD_Field(
        ...,
        title="Base Point Fraction",
        description="Fraction of points to use as base points",
    )


class IDWModalModel(ModalModelBase, IDWTransformConfig):
    """Inverse Distance Weighting (IDW) transformation."""

    type: ModalModelType = zCFD_Field("modal_model", description=_FSI_TYPE)
    transform_type: Literal["idw"] = zCFD_Field("idw", title="Transform Type")
    power: Union[float, List[float]] = zCFD_Field(
        ..., title="Power", description="Power parameters for IDW"
    )
    # Read unguarded by the FSI IDW morphing path, so they must always be
    # present.
    n_nearest: int = zCFD_Field(
        default=100,
        ge=1,
        title="N Nearest",
        description="Number of nearest points for IDW",
    )
    deformation_distance: float = zCFD_Field(
        default=1.0,
        gt=0,
        title="Deformation Distance",
        description="Maximum deformation distance",
    )
    blending_stiffness: float = zCFD_Field(
        default=0.5,
        ge=0,
        le=1,
        title="Blending Stiffness",
        description="Blending stiffness parameter",
    )


generic_fsi_transform_types = Literal["rbf_multiscale", "rbf", "idw"]
generic_fsi_basis_types = Literal["c4", "c2", "c0", "tps", "gaussian"]


class GenericFSIModel(BaseModel):
    """Generic iterative coupling FSI (loose external structural-solver coupling).

    Field set restored from the legacy voluptuous ``fsi_schema``
    (``zcfd_validate_legacy/utils.py``/``__init__.py`` ~1450-1470), which
    shares the RBF/IDW deformation knobs with the modal model but is keyed
    by an external coupling ``type`` rather than a Nastran modal basis.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["generic_fsi"] = zCFD_Field("generic_fsi", description=_FSI_TYPE)
    # Singular `zone` matches the key the C++ FSI zone lookup reads.
    zone: List[int] = zCFD_Field(
        ..., title="Zone", description="Zone IDs coupled to the external solver"
    )
    fixed_zones: Optional[List[int]] = zCFD_Field(
        default=None,
        title="Fixed Zones",
        description="Zones excluded from mesh deformation",
    )
    transform_type: generic_fsi_transform_types = zCFD_Field(
        ...,
        title="Transform Type",
        description="Mesh deformation transform used for the coupling",
    )
    support_radius: float = zCFD_Field(
        default=1.0, title="Support Radius", description="RBF support radius"
    )
    basis: generic_fsi_basis_types = zCFD_Field(
        default="c2", title="Basis", description="RBF basis function"
    )
    base_point_fraction: float = zCFD_Field(
        default=0.1,
        title="Base Point Fraction",
        description="Fraction of points used as RBF base points",
    )
    tol: float = zCFD_Field(
        default=0.01, title="Tol", description="RBF solve tolerance"
    )
    max_error: float = zCFD_Field(
        default=0.1,
        title="Max Error",
        description="Maximum allowed RBF interpolation error",
    )
    n_nearest: int = zCFD_Field(
        default=100,
        title="N Nearest",
        description="Number of nearest points for IDW",
    )
    deformation_distance: float = zCFD_Field(
        default=1.0,
        title="Deformation Distance",
        description="Maximum deformation distance",
    )
    blending_stiffness: float = zCFD_Field(
        default=0.5,
        title="Blending Stiffness",
        description="Blending stiffness parameter",
    )
    power: List[float] = zCFD_Field(
        default_factory=lambda: [3.0, 5.0, 10.0],
        title="Power",
        description="IDW power parameters",
    )
    user_variables: Optional[dict] = zCFD_Field(
        default=None,
        title="User Variables",
        description=(
            "Arbitrary variables passed to the external coupling solver "
            "(genuinely open-ended — not enumerable)."
        ),
    )


# Union of specific modal models
ModalModel = Annotated[
    Union[RBFModalModel, IDWModalModel], Field(discriminator="transform_type")
]


def default_fsi_type(v):
    """Before-validator: fill in an omitted 'type', which only a generic FSI model's 'zone' distinguishes."""
    if isinstance(v, dict) and "type" not in v:
        return {**v, "type": "generic_fsi" if "zone" in v else "modal_model"}
    return v


FSIModel = Annotated[
    Union[
        Annotated[ModalModel, Tag("modal_model")],
        Annotated[GenericFSIModel, Tag("generic_fsi")],
    ],
    Field(discriminator="type"),
]
