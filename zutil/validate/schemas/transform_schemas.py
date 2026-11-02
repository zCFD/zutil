from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    field_validator,
    model_validator,
)
from pydantic.json_schema import SkipJsonSchema
from typing import Annotated, Literal, Union, Optional, List
from zutil.validate.schemas.base_types import zCFD_Field
from .callables import PLACEMENT, PlacementFunction
from .motion_schemas import _IDLE_MOTION, ORDER, Motion
import numpy as np


class TransformsModel(BaseModel):
    """Placement and motion of the mesh."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    scale: Annotated[List[PositiveFloat], Field(min_length=3, max_length=3)] = (
        zCFD_Field(
            default_factory=lambda: [1.0, 1.0, 1.0],
            title="Scale",
            description=(
                "Factors that multiply the mesh file's x, y and z coordinates, for "
                f"example to convert inches to metres. {ORDER}"
            ),
        )
    )
    transform: Optional[
        Union[List[List[float]], np.ndarray, SkipJsonSchema[PlacementFunction]]
    ] = zCFD_Field(
        default=None,
        title="Transform",
        description=(
            "Static placement of the scaled mesh, applied once when the mesh is "
            "read: a 4x4 affine matrix whose translation part is in metres, or "
            + PLACEMENT.description("a function of the scaled node position")
            + f". {ORDER}"
        ),
    )
    motion: Annotated[Optional[Motion], _IDLE_MOTION] = zCFD_Field(
        default=None,
        title="Motion",
        description=(
            "Rigid motion of the placed mesh, or the frame its flow is solved in. "
            f"Its axis, origin and velocity are in placed coordinates. {ORDER}"
        ),
    )

    @field_validator("transform", mode="before")
    def validate_matrix_dims(cls, v):
        if v is None or callable(v):
            return v
        arr = np.array(v)
        if arr.shape != (4, 4):
            raise ValueError(
                "'transform' must be a 4x4 matrix or a function "
                f"{PLACEMENT.definition('place')}; it has shape {arr.shape}"
            )
        return arr


# RBF function categories
rbf_function_types = Literal[
    "tps",
    "thin_plate_spline",
    "multiquadric",
    "inverse_multiquadric",
    "gaussian",
    "linear",
    "cubic",
    "quintic",
    "c0",
    "c2",
    "c4",
    "c6",
]


class TransformBase(BaseModel):
    """Base configuration for transforms."""

    model_config = ConfigDict(extra="forbid")


class MeshDeformationConfig(TransformBase):
    """Base configuration for mesh deformation/interpolation."""

    # support_radius is common to both RBF and IDW (though IDW uses it as an alternative to stencil).
    # Read unguarded by the RBF morphing path,
    # so it must always be present; default matches ModalModelBase.support_radius / the legacy
    # modal "alpha" default (this defect class was already fixed there, see zutil/validate/schemas/fsi_schemas.py).
    support_radius: float = zCFD_Field(
        default=1.0,
        gt=0,
        title="Support Radius",
        description="Support radius for interpolation",
    )
    fixed_zones: Optional[List[int]] = zCFD_Field(
        default=None,
        title="Fixed Zones",
        description="Zones held in place while the moving zones deform.",
    )


class RBFTransformConfig(MeshDeformationConfig):
    """Configuration for Radial Basis Function (RBF) transforms.

    Multiscale RBF when base_point_fraction < 1.0 (default 0.1),
    full RBF when base_point_fraction=1.0.

    Deliberately excluded: the RBF morphing code
    still reads an optional boolean "use_local_rbf_nodes" that restricts each
    partition's RBF solve to only the nodes it actually owns, instead of the
    full global node set, as a memory/performance optimisation for large
    parallel RBF problems. It was evaluated for this schema and left out on
    purpose: no current control deck sets it, and every live RBF/IDW test
    case runs correctly using the full node set. The C++ read is guarded
    (defaults to using all nodes when the key is absent), so omitting the
    field here does not change behaviour for any existing deck.
    """

    # Default matches the C2 fallback RBFMorphBase::setBasisFunc applies
    # when the key is absent.
    basis: Optional[rbf_function_types] = zCFD_Field(
        default="c2",
        title="Basis",
        description="RBF function type",
    )
    base_point_fraction: float = zCFD_Field(
        default=0.2,
        gt=0,
        le=1,
        title="Base Point Fraction",
        description="Fraction of points to use as base points. "
        "1.0 = full RBF, < 1.0 = multiscale RBF.",
    )


class IDWTransformConfig(MeshDeformationConfig):
    """Configuration for Inverse Distance Weighting (IDW) transforms."""

    # Read unguarded by the IDW morphing path,
    # which unconditionally expects a 3-element list; must always be a List[float],
    # never a bare scalar. Default matches GenericFSIModel.power (fsi_schemas.py),
    # the FSI-side sibling of this field that already got this fix.
    power: List[float] = zCFD_Field(
        default_factory=lambda: [3.0, 5.0, 10.0],
        title="Power",
        description="IDW power parameters",
    )

    # IDW Specific: Stencil size (alternative to support_radius)
    stencil_size: Optional[int] = zCFD_Field(
        default=None,
        ge=1,
        title="Stencil Size",
        description="Maximum stencil size (number of nearest neighbors)",
    )

    # Read unguarded by the IDW morphing path, so it must
    # always be present; default matches the legacy default and IDWModalModel's
    # sibling field (fsi_schemas.py), where this defect class was already fixed.
    n_nearest: int = zCFD_Field(
        default=100,
        ge=1,
        title="N Nearest",
        description="Number of nearest points (legacy alias for stencil_size)",
    )

    # Read unguarded by the IDW morphing path,
    # so it must always be present; default matches the legacy default and
    # IDWModalModel's sibling field (fsi_schemas.py), where this defect class was
    # already fixed.
    deformation_distance: float = zCFD_Field(
        default=1.0,
        gt=0,
        title="Deformation Distance",
        description="Maximum deformation distance (IDW specific)",
    )
    blending_stiffness: Optional[float] = zCFD_Field(
        default=None,
        ge=0,
        le=1,
        title="Blending Stiffness",
        description="Blending stiffness parameter",
    )

    @model_validator(mode="after")
    def validate_idw_constraints(self):
        """Reject an ambiguous IDW stencil specification.

        support_radius and stencil_size are alternative ways of bounding
        the IDW stencil, so explicitly setting both is rejected as
        ambiguous - it's undefined which one would actually control the
        interpolation. `support_radius` inherits a required default
        (`MeshDeformationConfig.support_radius`, needed so the RBF sibling
        always has a value for its own unguarded C++ read) so it can no
        longer be tested via `is None`; `model_fields_set` is checked
        instead to see whether the caller actually passed it, as opposed
        to it merely taking its default.

        Leaving both unset (or setting neither explicitly) is accepted,
        not treated as the other XOR arm: the real IDW solver path
        reads only ``n_nearest`` from the dict, never
        ``support_radius`` or ``stencil_size`` - both the fluid-zone
        (``IDWTransformZone``) and FSI (``IDWModalModel``) subclasses are
        exercised by real control decks
        (``testcases/testcases/naca0012/n0012_449_a10p0_idw.py``,
        ``testcases/testcases/shockplatefsi/shockplatefsi_idw.py``) that
        set only ``n_nearest`` and never set either of these two fields.
        Requiring one of them to be set would reject those legitimate,
        currently-passing decks.
        """
        support_radius_explicit = "support_radius" in self.model_fields_set
        stencil_size_explicit = self.stencil_size is not None
        if support_radius_explicit and stencil_size_explicit:
            raise ValueError(
                "Cannot specify both 'support_radius' and 'stencil_size' - "
                "use only one to define the IDW stencil"
            )
        return self
