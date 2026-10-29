# ConvergenceControlBase

## cycles

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#cycles

## inner_convergence

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#inner-cycle-convergence

# CompressibleConvergenceControl

## scheme

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#scheme
demo: cylinder-convergence-scheme

## cfl

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#cfl-control

# IncompressibleMeshConvergenceControl

## viscosity_ramp

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#startup-ramps

## relaxation_ramp

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#startup-ramps

# IncompressibleConvergenceControl

## scheme

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#incompressible-convergence-control

# CFLModel

## cfl

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#cfl

## cfl_turbulence

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#cfl_turbulence

## cfl_coarse

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#cfl_coarse

## cfl_ramp

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#cfl_ramp

## cfl_viscous_factor

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#cfl_viscous_factor

# EulerScheme

## name

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#name

### euler

what: Explicit forward Euler in pseudo-time: one residual evaluation and update per cycle.
when: The `"euler"` scheme has a single stage and is the safest option for new or problematic cases.

## stage

fingerprint: pending

### 1

what: Under `runge kutta`, a single stage, the same update as `euler`. Other schemes ignore `stage`.

### 3

what: Under `runge kutta`, the three-stage TVD Runge-Kutta (SSP-RK3). Other schemes ignore `stage`.

### rk third order tvd

what: The same as `3`.

### 4

what: Under `runge kutta`, a four-stage scheme with stage coefficients 1/4, 1/2, 0.55 and 1. Other schemes ignore `stage`.

### 5

what: Under `runge kutta`, a five-stage scheme with stage coefficients 0.0695, 0.1602, 0.2898, 0.506 and 1. Other schemes ignore `stage`.

# RungeKuttaScheme

## name

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#name

### runge kutta

what: Explicit multistage Runge-Kutta in pseudo-time. `stage` selects the number of stages and their coefficients.

## stage

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#stage

# ImplicitEulerScheme

## name

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#implicit-euler

### implicit euler

what: The default implicit scheme, solving the full linearised system with the Petsc or AMGX linear solver each pseudo-time step.

# LUSGSScheme

## name

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#lu-sgs

### lu-sgs

what: A matrix-free, point-implicit LU-SGS/DP-LUR scheme. It avoids assembling and inverting the full linear system, trading some convergence rate for lower memory use and, often, faster wall-clock time per cycle than `"implicit euler"`.

## turbulence_solver

fingerprint: pending

### amg

what: Keeps the assembled AMG solve for the turbulence equations.

### point-implicit

what: Uses the matrix-free sweep instead.

# MFGMRESScheme

## name

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#mf-gmres

### mf-gmres

what: Restarted GMRES that applies the exact analytic flux Jacobian matrix-free, with no assembled mean-flow matrix.
when: Not available with the SAS turbulence model or the SAS hybrid.

## krylov subspace

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#mf-gmres

## preconditioner

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#mf-gmres

### block jacobi

what: Inverts the exact analytic 5x5 block diagonal. The same operator on the host and on a GPU.

### lu-sgs

what: Linearised symmetric Gauss-Seidel sweeps.

### lu-sgs nonlinear

what: The unlinearised LU-SGS sweep.
when: Requires `flexible: True`: the sweep is not a linear operator, and plain GMRES cannot converge with it.

### none

what: No preconditioner.

## orthogonalisation

fingerprint: pending

### mgs

what: Modified Gram-Schmidt.
when: The default.

### cgs2

what: Classical Gram-Schmidt with one reorthogonalisation pass and batched reductions: fewer reductions per restart cycle, at twice the vector work.
when: Only expected to pay at high rank counts, where each reduction is a real collective.

## turbulence solver

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#mf-gmres

### amg

what: Solves the assembled turbulence matrix with AMG.

### point-implicit

what: Solves the turbulence matrix with matrix-free sweeps.
when: The default, except where the backend forces `amg`.

## flexible

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#mf-gmres

# GTSRungeKuttaScheme

## name

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#global-time-stepping

### runge kutta

what: The three-stage TVD Runge-Kutta (SSP-RK3), third order in physical time.

## stage

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#global-time-stepping

### 3

what: The three-stage TVD Runge-Kutta (SSP-RK3), third order in physical time.

### rk third order tvd

what: The same as `3`.

# \_RampBase

## initial

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#cfl_ramp

## start_cycle

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#cfl_ramp

## end_cycle

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#cfl_ramp

# LinearRamp

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#cfl_ramp

### linear

what: Adds `increment` to the CFL every cycle between `start_cycle` and `end_cycle`.

# SteppedLinearRamp

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#cfl_ramp

### stepped_linear

what: Adds `increment` every `step_cycles` cycles.

# ExponentialRamp

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#cfl_ramp

### exponential

what: Multiplies the CFL by `factor` (>= 1.0) each cycle, capped at `max_allowed`.

# InnerConvergence

## target_orders

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#choosing-a-target

## target_ratio

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#choosing-a-target

## min_cycles

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#choosing-a-target

## variables

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#the-incompressible-measure

### mean flow

what: Density, momentum and energy for the compressible solver, momentum and pressure for the incompressible coupled scheme, and momentum alone for SIMPLE.

### all

what: Adds the turbulence transport equations.

## start_real_time_step

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#choosing-a-target

# CoupledScheme

## name

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#coupling-scheme

### coupled

what: Solves a single block-4 [u,v,w,p] system per pseudo-time step.

# SimpleScheme

## name

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#coupling-scheme

### simple

what: Uses a segregated SIMPLE momentum predictor followed by a pressure correction.

# StartupRamp

## initial

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#startup-ramps

## cycles

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#startup-ramps
