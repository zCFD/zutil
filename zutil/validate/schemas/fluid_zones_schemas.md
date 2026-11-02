# FluidZoneBase

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model

### momentum

what: Arbitrary momentum source supplied by a Python function.

### rotating

what: Rotating frame (MRF) on mesh cell zones of a still mesh.

### disc

what: Simple or blade-element (BET) turbine/propeller disc.

### porous

what: Darcy/Forchheimer momentum sink representing a porous obstruction.

### canopy

what: Vegetation drag/turbulence source for terrain wind modelling.

### abl

what: ABL-zone placeholder, not yet wired to a source term.

# MomentumSourceZone

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#momentum-source

## cell_zones

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#selecting-cells

## definition

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#selecting-cells

## func

fingerprint: pending
guide: /guide/setting-up/python-functions-in-the-deck#momentum-source-function

## kind

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#momentum-source

### fixed

what: Labels the source as constant; the solver evaluates `func` once at start-up whatever the kind.
when: The default; every kind gives the same result.

### ramp

what: Labels the source as ramped, but the solver does not read `kind`, so it behaves as `fixed`.
when: No effect; leave the default.

### func

what: Labels the source as function-driven, but the solver does not read `kind`, so it behaves as `fixed`.
when: No effect; leave the default.

# RotatingZone

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#rotating-zones

## cell_zones

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#selecting-cells

# TurbineController

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#actuator-discs

### tsr curve

what: A target tip-speed ratio driving a torque controller from a lookup curve; the user must specify a tip speed ratio curve.

### fixed

what: For a fixed controller the user must specify an angular velocity and blade pitch angle.

### schedule

what: Sets pitch and rotational speed by interpolating `pitch_schedule` and `speed_schedule` tables against the reference wind speed, and stops the rotor outside their range.
when: Not available from a validated deck, as the schema does not accept `pitch_schedule` or `speed_schedule`.

# TurbineGeometry

## normal

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#actuator-discs

# TurbineModel

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#actuator-discs

### simple

what: A simple turbine model, which averages the performance of the turbine over the entire disc, with fixed or curve-driven thrust/power coefficients.

### bet

what: A blade element model, which adds the blade/aerofoil description.

## kind

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#actuator-discs

### propellor

what: Treats the disc as a propeller that adds momentum to the flow; with a `simple` fixed model, thrust and power come from `thrust_coefficient` and `power_coefficient` scaled by rotational speed and diameter.
when: For propellers and other powered rotors.

### turbine

what: Treats the disc as a wind turbine that extracts momentum from the flow; with a `simple` fixed model, thrust comes from `thrust_coefficient` on the swept area and power from `power`.
when: For wind and tidal turbines.

## power model

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#actuator-discs

### fixed

what: Uses a single operating point: `thrust_coefficient` with `power` for a turbine, or with `power_coefficient` for a propeller.
when: For a `simple` model at one operating condition; `kind` must also be set.

### curve

what: Interpolates power from `power_curve` and thrust coefficient from `thrust_coefficient_curve` against the reference wind speed, and gives zero outside the tabulated range.
when: For a `simple` turbine model across a range of wind speeds; `kind` is not read.

## tip loss correction

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#actuator-discs

### none

what: Applies no tip-loss correction.
when: Equivalent to leaving the key unset.

### elliptic

what: Scales sectional lift and drag by √(1 − r²), where r is the radius as a fraction of `outer_radius`.
when: A `bet` model that needs a simple tip loss across the whole span.

### acos-fit

what: Scales sectional lift and drag by an arccos fit of the Prandtl factor, which is close to one inboard and falls to zero at the tip.
when: A `bet` model that needs the loss confined to the outer span.

### acos shift-fit

what: Intended as a shifted arccos fit, but the blade element model matches only `acos_shift-fit`, so this value applies no correction.
when: No effect at present; use `acos-fit`.

### f-fit

what: Scales sectional lift and drag by an empirical fit that departs from one only close to the tip.
when: A `bet` model that needs a correction limited to the last few per cent of span.

### rstar

what: Leaves sections inboard of `tip_loss_correction_radius` uncorrected and scales those outboard by an elliptic factor that falls to zero at the tip.
when: A `bet` model where the loss should start at a chosen fraction of `outer_radius`.

# ActuatorDiscZone

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#actuator-discs

## definition

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#selecting-cells

# PorousMediaZone

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#porous-media

## cell_zones

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#selecting-cells

## definition

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#selecting-cells

## alpha

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#porous-media

## C2

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#porous-media

# CanopyZone

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#canopy

## cell_zones

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#selecting-cells

## definition

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#selecting-cells

## field

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#canopy

## function

fingerprint: pending
guide: /guide/setting-up/python-functions-in-the-deck#leaf-area-density-function

# ABLZone

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-fluid-zone-model#atmospheric-boundary-layer-abl

# TransformZoneBase

## func

fingerprint: pending
guide: /guide/setting-up/python-functions-in-the-deck#mesh-deformation-function

# IDWTransformZone

## type

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#mesh-deformation

# RBFTransformZone

## type

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#mesh-deformation
