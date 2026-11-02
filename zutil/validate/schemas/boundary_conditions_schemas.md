# BoundaryConditionBase

## type

fingerprint: f207638a5597
guide: /guide/choosing/choosing-boundary-conditions

## zones

fingerprint: 5bc18e9f3aa6
guide: /guide/choosing/choosing-boundary-conditions#zones

# RoughnessModel

## type

fingerprint: ebdfb663ee2f
guide: /guide/choosing/choosing-a-wall-model#roughness

## thermal model

fingerprint: c454da7992a1
guide: /guide/choosing/choosing-a-wall-model#rough-compressible-walls

# WallBoundary

## kind

fingerprint: 68546b9d604f
guide: /guide/choosing/choosing-a-wall-model#kind
demo: plate-wall-kind

### no slip

when: Viscous wall with the mesh resolved to $y^+ \sim 1$.

### wall function

when: Viscous wall with the first cell between $y^+ = 30$ and $250$; it tolerates a first cell down to $y^+ \sim 1$. The law of the wall (Spalding's with SST, the Allmaras fit with SA-neg) represents a flat plate in zero pressure gradient, so accuracy falls on highly curved surfaces and in strong pressure gradients.

## temperature

fingerprint: fa9c3d89f588
guide: /guide/choosing/choosing-a-wall-model#wall-function
demo: plate-wall-temperature

## heat flux

fingerprint: 27d473b887cd
guide: /guide/choosing/choosing-a-wall-model#wall-function

## compressible wall function

fingerprint: f8ed21c2af7b
guide: /guide/choosing/choosing-a-wall-model#wall-function
demo: plate-wall-compressible-wall-function

## roughness

fingerprint: b16d834979b4
guide: /guide/choosing/choosing-a-wall-model#roughness
demo: plate-wall-roughness

## fsi

fingerprint: c6fa6a37abf1
guide: /guide/working-with/fluid-structure-interaction

# OversetConnection

## model

fingerprint: 70ff810119f5
guide: /guide/working-with/overset-meshes#connection

# OversetInterface

## connection

fingerprint: b16d834979b4
guide: /guide/working-with/overset-meshes#connection

## interpolation method

fingerprint: f3715a31cdbe
guide: /guide/working-with/overset-meshes#interpolation_method

### least-squares

when: On a moving interface, where only it and `green-gauss` reproduce a field with a gradient at the receptor point.

## additional active layers

fingerprint: 9fd819f4f5ce
guide: /guide/working-with/overset-meshes#additional_active_layers

## blank enclosed cells

fingerprint: 05ddd00c4f22
guide: /guide/working-with/overset-meshes#blank_enclosed_cells

## moving mesh halo layers

fingerprint: 9fd819f4f5ce
guide: /guide/working-with/overset-meshes#moving_mesh_halo_layers

# SlidingConnection

## model

fingerprint: cdd80e3b2e36
guide: /guide/choosing/choosing-boundary-conditions#sliding

## boundary

fingerprint: cdd80e3b2e36
guide: /guide/choosing/choosing-boundary-conditions#sliding

# SlidingInterface

## connection

fingerprint: b16d834979b4
guide: /guide/choosing/choosing-boundary-conditions#sliding

## area

fingerprint: be08c585e538
guide: /guide/choosing/choosing-boundary-conditions#sliding

## flux

fingerprint: 299f8c848230
guide: /guide/choosing/choosing-boundary-conditions#sliding

# PeriodicConnection

## boundary

fingerprint: cdd80e3b2e36
guide: /guide/choosing/choosing-boundary-conditions#periodic

# PeriodicInterface

## connection

fingerprint: b16d834979b4
guide: /guide/choosing/choosing-boundary-conditions#periodic

## transform

fingerprint: c6fa6a37abf1
guide: /guide/choosing/choosing-boundary-conditions#periodic

# FarfieldBoundary

## kind

fingerprint: 5124ca0150e5
guide: /guide/choosing/choosing-boundary-conditions#farfield

## target

fingerprint: b16d834979b4
guide: /guide/working-with/trimming-to-a-target

# InflowBoundaryBase

## type

fingerprint: 5906483590af
guide: /guide/choosing/choosing-boundary-conditions#inflow

# OutflowBoundaryBase

## type

fingerprint: b589a9f16ab4
guide: /guide/choosing/choosing-boundary-conditions#outflow

# PressureOutflowBoundary

## kind

fingerprint: pending
guide: /guide/choosing/choosing-boundary-conditions#outflow
demo: pipe-outflow-kind
