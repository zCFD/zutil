# SolutionModel

## format

fingerprint: pending
guide: /guide/working-with/post-processing-and-visualisation

### none

what: Outputs no files.

### vtk

what: Outputs in VTK HDF format (.vtkhdf) for reading into [ParaView](https://www.paraview.org). The .vtkhdf output format requires [ParaView 6.1.1](https://www.paraview.org/download/) or later.

### ensight

what: Writes the volume solution as EnSight files with VTK's EnSight writer, one set per output cycle. Surface output is still written as VTK HDF.
when: For post-processing in EnSight.

### fwensight

what: Writes the same VTK HDF output as `vtk`, and in addition an EnSight Gold copy (`.encas` and `.geo` files) of the volume and of the boundaries.

### native

what: Writes each volume variable to its own HDF5 file per output cycle, with an index file of external links to them. Surface output is still written as VTK HDF.

## fwh_interpolate

fingerprint: pending
guide: /guide/working-with/aeroacoustics-fwh

## fwh_wall_data

fingerprint: pending
guide: /guide/working-with/aeroacoustics-fwh

## precision

fingerprint: pending

### single

what: 32-bit floating-point field values. The solver writes field data at this precision.
when: The default.

### double

what: Accepted, but not read by the solver: field data is written in single precision whatever is set.

## write_restart

fingerprint: pending
guide: /guide/choosing/choosing-initialisation-and-reference-state#restart

# ForceModel

## transform

fingerprint: pending
guide: /guide/setting-up/python-functions-in-the-deck#transform-function
