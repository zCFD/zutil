parameters = {
    # units for dimensional quantities
    "units": "SI",
    # reference state
    "reference": "IC_1",
    #'restart' : 'true',
    "time marching": {
        "unsteady": {
            "total time": 2.0,
            "time step": 0.002,
            "order": "second",
            "start": 3000,
        },
        "cycles": 5,
    },
    "linear solver options": {
        "flow": {
            "-ksp_type": "fgmres",
            "-ksp_rtol": "1.0e-8",
            "-ksp_atol": "1.0e-16",
            "-ksp_max_it": "20",
            "-ksp_min_it": "2",
            "-ksp_converged_reason": "",
            "-ksp_monitor": "",
        },
        # "-pc_type": "hypre",
        # "-pc_hypre_type": "boomeramg"},
    },
    "equations": "incomp viscous",
    "incomp viscous": {
        "order": "second",
        "precondition": True,
        "relax": 0.5,
    },
    "material": "air",
    "air": {
        "gamma": 1.4,
        "gas constant": 287.0,
        "Sutherlands const": 110.4,
        "Prandtl No": 0.72,
        "Turbulent Prandtl No": 0.9,
    },
    "IC_1": {
        "temperature": 277.7777,
        "pressure": 101325.0,
        "V": {
            "vector": [0.0, 334.35 * 0.2, 0.0],
            "Mach": 0.2,
        },
        "Reynolds No": 150.0,
        "Reference Length": 1.0,
        "turbulence intensity": 1.0e-4,
        "eddy viscosity ratio": 0.1,
    },
    "BC_1": {
        "ref": 7,
        "type": "symmetry",
    },
    "BC_2": {
        "ref": 3,
        "type": "wall",
        "kind": "noslip",
    },
    "BC_3": {
        "zone": [1],
        "type": "inflow",
        "condition": "IC_1",
        "kind": "default",
    },
    "BC_4": {
        "zone": [4],
        "type": "outflow",
        "condition": "IC_1",
        "kind": "pressure",
    },
    "report": {
        "frequency": 5,
        "monitor": {
            "MR_1": {
                "name": "probe",
                "point": [0.25, 1.07, 0.313],
                "variables": ["cp"],
            },
        },
    },
    "write output": {
        "format": "vtk",
        "surface variables": ["V", "p", "T", "rho", "m", "cp"],
        "volume variables": ["V", "p", "T", "rho", "m", "cp"],
        "frequency": 100,
    },
}
