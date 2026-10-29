# EquationsBase

## type

fingerprint: pending
guide: /guide/choosing/choosing-an-equation-set#equation-sets

### euler

what: Use the Euler flow equations in the simulation. These are compressible, inviscid flow equations.

### viscous

what: Use the viscous flow equations. These are compressible, viscid flow equations which do not model turbulence.

### rans

what: Use the RANS (Reynolds-Averaged Navier-Stokes) equations in the simulation. These are compressible, viscid flow equations with additional terms included to account for the effect of turbulence on the flow without the expense of simulating the turbulent flow structures themselves.

### les

what: Use the Large Eddy Simulation (LES) equations.

## precondition

fingerprint: pending
guide: /guide/choosing/choosing-an-equation-set#precondition

# RANSEquations

## turbulence

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#turbulence-dictionary

# LESEquations

## turbulence

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#les

# TurbulenceModel

## model

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model

### sst

what: The Menter Shear Stress Transport Turbulence Model (https://turbmodels.larc.nasa.gov/sst.html)

### spalart-allmaras

what: Spalart-Allmaras turbulence model (https://turbmodels.larc.nasa.gov/spalart.html)

### sa-neg

what: Negative Spalart-Allmaras turbulence model (https://turbmodels.larc.nasa.gov/spalart.html)

### sas

what: Scale-Adaptive Simulation, run as a standalone RANS model

### sst-transition

what: The Langtry-Menter 4-equation Transitional SST Model (https://turbmodels.larc.nasa.gov/langtrymenter_4eqn.html)

## hybrid

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#hybrid

### DES

what: Detached Eddy Simulation

### DDES

what: Delayed Detached Eddy Simulation

### IDDES

what: Improved Delayed Detached Eddy Simulation

### SAS

what: Scale-Adaptive Simulation run as the hybrid branch of `model`

## qcr

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#qcr

### 2000

what: [QCR 2000](<https://doi.org/10.1016/S0142-727X(00)00007-2>)

### 2020

what: [QCR 2020](https://doi.org/10.2514/1.J059683)

### 2021

what: [QCR 2021](https://doi.org/10.1017/aer.2021.42)

## production

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#production

### vorticity

what: The vorticity based production term (SST-V) $P_m=\mu_t\Omega_v^2-\frac{2}{3}\rho k \delta_{ij}\frac{\partial u_i}{\partial x_j}$

### strain-incompressible

what: The strain based production term for incompressible flows $P_m=\mu_t S^2$

### strain-compressible

what: The strain based production term for compressible flows $P_m=\mu_t S^2 -\frac{2}{3}\rho k \delta_{ij}\frac{\partial u_i}{\partial x_j}$

### kato-launder

what: The Kato-Launder source term $P_m=\mu_t S\Omega_v -\frac{2}{3}\rho k \delta_{ij}\frac{\partial u_i}{\partial x_j}$

## betastar

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## cdes_kw

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## cdes_keps

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## cd1

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## cd2

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## cw

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## a1

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## cdes

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## ca1

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## ca2

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## ce1

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## ce2

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## cthetat

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## sigmagamma

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## sigmathetat

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## separation_correction

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#advanced-parameters-for-turbulence-models

## les_cfl_limit

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#shielding-the-les-branch-by-cfl

## les_cfl_width

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#shielding-the-les-branch-by-cfl

## les_turnover_coefficient

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#les_turnover_coefficient

# LESModel

## les

fingerprint: pending
guide: /guide/choosing/choosing-a-turbulence-model#les

### none

what: Implicit LES (no explicit SGS model).

### wale

what: The Wall-Adapting Local Eddy-viscosity model.

### deardorff

what: The Deardorff 1.5-order closure, which transports the subgrid turbulent kinetic energy $e$ and limits the mixing length under stable stratification. The eddy viscosity is $0.1\,l\sqrt{e}$ and the heat diffusivity is $(1 + 2l/\Delta)$ times it.
when: For stratified atmospheric boundary layers with the incompressible solver. The compressible solver does not implement it.

# IncompressibleEquationsOptions

## correction

fingerprint: pending
guide: /guide/choosing/choosing-an-equation-set#incompressible-viscous

### none

what: No non-orthogonal correction. The implicit face-normal gradient uses the full face area vector and the explicit correction is zero.
when: Only for meshes that are close to orthogonal.

### orthogonal

what: Splits the face area vector $\mathbf{S}$ into an implicit part $\boldsymbol{\Delta} = |\mathbf{S}|\,\hat{\mathbf{d}}$ along the line between cell centres, with the remainder treated explicitly.

### minimum

what: Takes the implicit part as the projection of $\mathbf{S}$ on the line between cell centres, $\boldsymbol{\Delta} = (\mathbf{d}\cdot\mathbf{S}/\mathbf{d}\cdot\mathbf{d})\,\mathbf{d}$, which keeps the explicit remainder smallest.

### over_relaxed

what: Takes the implicit part as $\boldsymbol{\Delta} = \mathbf{d}\,|\mathbf{S}|^2/(\mathbf{d}\cdot\mathbf{S})$, which grows as the mesh becomes more non-orthogonal.
when: The default.

## gradient_limiter_type

fingerprint: pending

### cellmd

what: More robust on highly non-orthogonal meshes.

### cell

what: Less dissipative.

## energy_variable

fingerprint: pending

### temperature

what: The energy equation transports static temperature $T$.
when: The default, for heat transfer without stratification.

### potential temperature

what: Potential temperature $\theta = T (p_0/p)^{R/c_p}$.
when: The natural choice for atmospheric cases.

# IsothermalBaseState

## kind

fingerprint: pending

### isothermal

what: Uniform temperature $T_0$, so potential temperature grows with height as $\theta = \theta_0 \exp(g z / (c_p T_0))$.

# ConstantLapseRateBaseState

## kind

fingerprint: pending

### constant lapse rate

what: Temperature falls linearly with height, $T = T_0 - \Gamma z$, with $\Gamma$ set by `lapse_rate`.

# TwoLayerBaseState

## kind

fingerprint: pending

### two layer

what: A neutral mixed layer of uniform $\theta_0$ up to `mixed_layer_depth`, capped by stable stratification in which $\theta$ rises at `upper_lapse_rate`.
when: The GABLS1 stable boundary layer set-up.

# ConstantBruntVaisalaBaseState

## kind

fingerprint: pending

### constant brunt vaisala

what: Uniform stratification $\theta = \theta_0 \exp(N^2 z / g)$.

# ThetaPerturbation

## kind

fingerprint: pending

### inertia gravity wave

what: $\theta' = A \sin(\pi z/H) / (1 + ((x-x_c)/a)^2)$ (Skamarock & Klemp 1994).

### bubble

what: $\theta' = A \cos^2(\pi r/2)$ inside the ellipse $r = |((x-x_c)/r_x, (z-z_c)/r_z)| \le 1$, zero outside.

### random

what: A pseudo-random $\theta'$ of up to `amplitude` either side of zero, hashed from each cell's position and applied only below `depth`. The seed is reproducible and independent of the partitioning.
when: Seeding turbulence in a stable boundary layer, as in GABLS1.

# VelocityPerturbation

## kind

fingerprint: pending

### random

what: Independent pseudo-random perturbations of up to `amplitude` either side of zero on each velocity component, hashed from each cell's position and applied only below `depth`. The seed is reproducible and independent of the partitioning.
