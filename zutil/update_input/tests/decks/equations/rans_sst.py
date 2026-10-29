parameters = {
    "equations": "RANS",
    "RANS": {
        "order": "second",
        "limiter": "van albada",
        "precondition": True,
        "preconditioner": {"minimum mach number": 0.01},
        "turbulence": {
            "model": "sst",
            "les": "none",
            "betastar": 0.09,
            "limit mut": True,
            "limit gradient k": 0.5,
            "qcr": 2000,
            "rotation correction": True,
            "production": 1,
            "separation correction": False,
            "cdes_kw": 0.78,
            "a1": 0.31,
        },
    },
    "material": "air",
    "air": {"gamma": 1.4, "Turbulent Prandtl No": 0.9},
}
