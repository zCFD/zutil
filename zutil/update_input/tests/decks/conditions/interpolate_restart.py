parameters = {
    "equations": "RANS",
    "reference": "IC_1",
    "restart": True,
    "interpolate restart": True,
    "restart meshname": "wing_coarse",
    "restart casename": "wing_coarse_run",
    "solution smooth cycles": 5,
    "IC_1": {
        "temperature": 288.15,
        "pressure": 101325.0,
        "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.2},
        "Reynolds No": 6.0e6,
    },
}
