parameters = {
    "equations": "RANS",
    "reference": "IC_1",
    "IC_1": {
        "temperature": 288.15,
        "pressure": 101325.0,
        "V": {"vector": [10.0, 0.0, 0.0]},
        "viscosity": 1.8e-5,
        "profile": {
            "ABL": {
                "roughness length": 0.0003,
                "friction velocity": 0.4,
                "surface layer height": 100.0,
                "Monin-Obukhov length": -50.0,
                "tke": 3.3,
                "z0": 0.0,
                "up": [0.0, 0.0, 1.0],
            }
        },
    },
}
