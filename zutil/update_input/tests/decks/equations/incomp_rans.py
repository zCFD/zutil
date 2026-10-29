parameters = {
    "equations": "incomp rans",
    "incomp rans": {
        "order": "second",
        "relax": 0.7,
        "correction": "over relaxed",
        "correction limiter": 0.4,
        "gradient limiter type": "cell",
        "gradient limiter": 0.6,
        "max second order non orthogonal angle": 80.0,
        "central blend": 0.9,
        "inviscid flux scheme": "HLLC",
        "turbulence": {"model": "sst", "les": "none", "relax": 0.3},
    },
    "material": "air",
    "air": {"gravity": [0.0, 0.0, -9.81]},
}
