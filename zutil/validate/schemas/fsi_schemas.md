# ModalModelBase

## file format

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#modal-model

### nastran

what: Reads the structural grid from `<nastran_casename>.bdf` and the mode shapes and frequencies from `<nastran_casename>.op2`.
when: Always; it is the only format supported.

## harmonic_initial_state

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#harmonic-initial-state

# RBFModalModel

## type

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#modal-model

### modal_model

what: Selects the built-in Nastran modal-model coupling.

## transform type

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#rbf-multiscale-transform

### rbf_multiscale

what: Deforms the volume mesh using the multiscale RBF scheme.

# IDWModalModel

## type

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#modal-model

### modal_model

what: Selects the built-in Nastran modal-model coupling.

## transform type

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#idw-transform

### idw

what: Deforms the volume mesh using inverse distance weighting.

# GenericFSIModel

## type

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#generic-fsi

### generic_fsi

what: Selects the generic external-solver coupling.

## transform type

fingerprint: pending
guide: /guide/working-with/fluid-structure-interaction#interpolation-method

### rbf

what: Deforms the volume mesh with a single-level RBF, adding surface points one at a time, worst-interpolated first, until each is within `tol`.
when: Where an error-controlled choice of points is preferred to the fixed `base_point_fraction` of `rbf_multiscale`.
