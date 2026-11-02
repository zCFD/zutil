parameters = {
    "equations": "RANS",
    "reference": "IC_1",
    "IC_1": {
        "temperature": 288.15,
        "pressure": 101325.0,
        "V": {"vector": [10.0, 0.0, 0.0]},
        "viscosity": 1.8e-5,
        "profile": {"field": "inflow_profile.vtp", "use wall distance": True},
    },
}
