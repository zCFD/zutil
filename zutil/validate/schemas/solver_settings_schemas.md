# SolverSettingsPropertiesModel

## type

fingerprint: pending
guide: /guide/choosing/choosing-an-equation-set#equation-sets

## linear_solver_options

fingerprint: pending
guide: /guide/choosing/choosing-a-linear-solver

## solution_smooth_cycles

fingerprint: pending
guide: /guide/choosing/choosing-initialisation-and-reference-state#restart

# LinearSolverOptions

## flow

fingerprint: pending
guide: /guide/choosing/choosing-a-linear-solver#petsc

## turbulence

fingerprint: pending
guide: /guide/choosing/choosing-a-linear-solver#petsc

## rbf

fingerprint: pending
guide: /guide/choosing/choosing-a-linear-solver#petsc

## double_precision

fingerprint: pending
guide: /guide/choosing/choosing-a-linear-solver#amgx

# IncompressibleSolverSettingsPropertiesModel

## linear_solver_options

fingerprint: pending
guide: /guide/choosing/choosing-a-linear-solver#incompressible-solver

# IncompressibleLinearSolverOptions

## flow

fingerprint: pending
guide: /guide/choosing/choosing-a-linear-solver#incompressible-solver

## turbulence

fingerprint: pending
guide: /guide/choosing/choosing-a-linear-solver#incompressible-solver

## rbf

fingerprint: pending
guide: /guide/choosing/choosing-a-linear-solver#incompressible-solver

## double_precision

fingerprint: pending
guide: /guide/choosing/choosing-a-linear-solver#incompressible-solver

## pressure_correction

fingerprint: pending
guide: /guide/choosing/choosing-a-linear-solver#incompressible-solver

## momentum

fingerprint: pending
guide: /guide/choosing/choosing-a-linear-solver#incompressible-solver

# CompressibleSolverSettingsPropertiesModel

## type

fingerprint: pending

### compressible

what: The density-based solver for the compressible equations: `euler`, `viscous`, `rans` or `les` with an ideal gas.
when: Where density varies with pressure and temperature, including transonic and supersonic flow.

# IncompressibleSolverSettingsPropertiesModel

## type

fingerprint: pending

### incompressible

what: The pressure-based solver for incompressible flow, coupling momentum and pressure with either the `coupled` or the `simple` scheme.
when: Low-speed flow in which density does not depend on pressure. Buoyancy and anelastic stratification are available through the energy equation.

# CellReorderingOptions

## sfc_curve

fingerprint: pending

### hilbert

what: Orders each partition's interior cells along a Hilbert curve through their centroids. The curve has no jumps, so cells close in the ordering are close in space.
when: The default. The ordering is kept only if it reduces the cell-adjacency profile.

### morton

what: Orders each partition's interior cells along a Morton (Z-order) curve, which interleaves the bits of the centroid coordinates. The curve jumps between distant regions.
when: An alternative to `hilbert`. It too is kept only if it reduces the cell-adjacency profile.
