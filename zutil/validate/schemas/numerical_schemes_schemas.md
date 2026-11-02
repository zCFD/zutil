# FVNumericalSchemeBase

## limiter

fingerprint: pending
guide: /guide/choosing/choosing-a-limiter
demo: shocktube-limiter

### van albada

what: Default. One-dimensional: $\psi = \max\left(0, \dfrac{2ab + \varepsilon^2}{a^2 + b^2 + \varepsilon^2}\right)$. Smooth, symmetric, bounded in $[0, 1]$.
when: The recommended general purpose choice.

### van leer

what: One-dimensional: $\psi = \max\left(0, \dfrac{4ab + \varepsilon^2}{(a + b)^2 + \varepsilon^2}\right)$. Slightly more compressive than van Albada.

### symmetric ratio

what: One-dimensional: $\psi = \max\left(0, \min\left(1, \dfrac{4\min\left(\sigma a, \sigma b\right) + \varepsilon}{\left|a + b\right| + \varepsilon}\right)\right)$ with $\sigma = \mathrm{sign}(a + b)$. The least dissipative of the three.

### none

what: $\psi = 1$. Unlimited reconstruction: second order everywhere, but not monotonicity preserving.
when: Appropriate only for smooth flows with no shocks or strong gradients.

### barth jespersen

what: Cell-based: $P(y) = \min(1, y)$. Barth & Jespersen (1989) as published. Bound preserving, but only $C^0$: the kink at $y = 1$ makes the limiter value flicker between cycles and can stall the residual. It has no smoothness deactivation, so it limits wherever the bound binds.

::: {.warning}

`"barth jespersen"` is the multidimensional, cell-based Barth & Jespersen limiter. It is not the one-dimensional `"symmetric ratio"` function, which takes no minimum or maximum over the cell's neighbours. A deck whose results depend on the one-dimensional behaviour must set `"symmetric ratio"`; `"barth jespersen"` gives different results. See the [Unreleased](/release-notes/unreleased) release notes for when this changed.

:::

### michalak gooch

what: Cell-based: $P(y) = y - \tfrac{4}{27}y^3$ for $y < 3/2$, else 1. Michalak & Ollivier-Gooch. $C^1$, and $P(y) \leq \min(1,y)$, so never less monotonicity preserving than Barth-Jespersen. Deactivates in smooth flow via a grid-scaled threshold, so formal order survives refinement.
when: The recommended choice.

### venkatakrishnan

what: Cell-based: $P(y) = \min\left(1, (y^2+2y)/(y^2+y+2)\right)$. Smooth, but exceeds 1 for $y > 2$, so it needs the clip, which reintroduces a $C^0$ kink at $y = 2$.
when: Provided as a familiar cross-check.

## freeze_limiter_cycle

fingerprint: pending
guide: /guide/troubleshooting/troubleshooting#limiter-switching

## inviscid flux scheme

fingerprint: pending
guide: /guide/choosing/choosing-a-spatial-scheme#choosing-the-flux-scheme
demo: shocktube-inviscid-flux-scheme

### roe

what: Roe's approximate Riemann solver. `entropy_fix` floors each wave speed at that fraction of the largest. Read by the compressible solver only.
when: The default.

### hllc

what: The HLLC approximate Riemann solver, which restores the contact wave missing from HLL.

### rusanov

what: The Rusanov (local Lax-Friedrichs) flux, which scales the dissipation by the largest local wave speed.
when: The most dissipative of the schemes.

### roe low-diss

what: The Roe flux with its upwind dissipation scaled by a factor between `roe_dissipation_sensor_minimum` and 1, set per face by `roe_low_dissipation_sensor`.
when: Scale-resolving simulations, where full upwind dissipation would damp the resolved turbulence. With the sensor `NONE` it is the same as `roe`.

### hllc low-diss

what: The HLLC flux with its dissipation scaled in the same way as `roe low-diss`.

## order

fingerprint: pending
guide: /guide/choosing/choosing-a-spatial-scheme#spatial-order

### first

what: First-order upwind: face values are the upwind cell values, with no reconstruction.

### second

what: Second order. The compressible solver reconstructs every variable to the face with the limited scheme set by `reconstruction`; the incompressible solver uses linear-upwind interpolation for momentum.
when: The default.

### euler_second

what: The compressible solver reconstructs the mean-flow variables to second order and keeps the turbulence variables first order. The incompressible solver blends 75% linear and 25% linear-upwind interpolation for momentum.

### central

what: Incompressible solver only. Momentum is interpolated by central differencing blended with linear-upwind, in the proportion set by `central_blend`.

## gradient

fingerprint: pending

### green-gauss

what: Green-Gauss gradients from face values interpolated by inverse distance, or by a plain average when `linear_gradients` is set.
when: The default.

### least-squares

what: A least-squares fit to the differences between each cell and its face neighbours, weighted by face area.

## reconstruction

fingerprint: pending

### muscl

what: The MUSCL $\kappa$-scheme with $\kappa = 1/3$, blending the limited cell-gradient extrapolation with the central difference across the face.
when: The default.

### umuscl

what: The unstructured MUSCL of Burg, $u_f = u_L + (1-\kappa)\,\mathbf{r}\cdot\nabla u_L + \tfrac{\kappa}{2}(u_R - u_L)$ with $\kappa = 1/3$, scaled by the limiter.

### lpumuscl

what: Accepted, but the solver currently treats it as `muscl`.

# FVCompressibleNumericalSchemeBase

## order

fingerprint: pending
guide: /guide/choosing/choosing-a-spatial-scheme#spatial-order
demo: shocktube-order

## multigrid

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#multigrid

## roe_low_dissipation_sensor

fingerprint: pending

### NONE

what: No scaling: the low-dissipation fluxes apply full upwind dissipation.
when: The default.

### FD

what: Scales the dissipation by $1 - f_d$, where $f_d$ is the DDES shielding function of the turbulence model: full dissipation in RANS regions, the minimum in LES regions. Faces the shock sensor flags keep full dissipation.
when: With a DES-family hybrid. Without a RANS turbulence model every cell is treated as LES.

### DES

what: Scales the dissipation by the turbulence model's length-scale ratio, capped at 1, which is near 0 in LES regions and 1 in RANS regions. Faces the shock sensor flags keep full dissipation.
when: With a DES-family hybrid. Without a RANS turbulence model every cell is treated as LES.

### NTS

what: The blending function of Xiao et al. (2012), built from vorticity, strain rate, the turbulent length scale and the cell size, and combined with the shock sensor.

## roe_low_dissipation_sensor_parameter

fingerprint: pending

### density

what: The shock sensor is the normalised density jump between each cell and its face neighbours.
when: The default when unset.

### velocity

what: The shock sensor is the Ducros ratio $|\nabla\cdot\mathbf{u}| / (|\nabla\cdot\mathbf{u}| + |\boldsymbol{\omega}|)$.

### velocity divergence/vorticity

what: The same as `velocity`.

# MultigridConfig

## levels

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#multigrid

## cycles

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#multigrid

# FVCompressibleRANSNumericalScheme

## turbulence_limiter

fingerprint: pending

### no update

what: When an update leaves a cell's turbulence variables invalid, the cell keeps its turbulence values from before the update.
when: The default when unset.

### zero

what: When an update leaves a cell's turbulence variables invalid, they are clipped to their lower limits.
