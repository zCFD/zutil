# MeshDeformationConfig

## support radius

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#interpolation-method

# RBFTransformConfig

## basis

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#interpolation-method

### c2

what: A Wendland C2 function that is zero beyond `support_radius`, so only nearby points interact and the system is sparse.
when: The default, and the appropriate choice for most cases.

### c0

what: A Wendland C0 function that is zero beyond `support_radius`; the deformation is continuous but its slope is not.
when: Rarely; `c2` gives a smoother deformation with the same sparsity.

### c4

what: Selects a Wendland C4 function, but the matrix assembly and the volume deformation evaluate `c2`, so the interpolation is inconsistent.
when: Not at present; use `c2`.

### gaussian

what: A Gaussian whose width is `support_radius`; it is never zero, so every pair of points interacts and the system is dense.
when: Small surfaces only, as the dense system grows with the square of the number of points.

### tps

what: A thin-plate spline in the distance scaled by `support_radius`; it is never zero, so the system is dense.
when: Small surfaces where a global, smooth deformation is wanted.

### thin_plate_spline

what: Not implemented under this name; the run stops with an unknown-basis error.
when: Never; the thin-plate spline is `tps`.

### multiquadric

what: Not implemented; the run stops with an unknown-basis error.
when: Never; use `c2`.

### inverse_multiquadric

what: Not implemented; the run stops with an unknown-basis error.
when: Never; use `c2`.

### linear

what: Not implemented; the run stops with an unknown-basis error.
when: Never; use `c2`.

### cubic

what: Not implemented; the run stops with an unknown-basis error.
when: Never; use `c2`.

### quintic

what: Not implemented; the run stops with an unknown-basis error.
when: Never; use `c2`.

### c6

what: Not implemented; the run stops with an unknown-basis error.
when: Never; use `c2`.

## base point fraction

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#interpolation-method

# IDWTransformConfig

## power

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#interpolation-method
