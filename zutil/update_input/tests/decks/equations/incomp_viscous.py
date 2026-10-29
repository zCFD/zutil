parameters = {
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
}
