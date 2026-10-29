mach = 0.2


parameters = {
    "reference": "IC_1",
    "material": "air",
    "air": {"gamma": 1.4, "gas constant": 287.0},
    "IC_1": {"temperature": 300.0, "pressure": 101325.0},
    "FZ_1": {
        "type": "translating",
        "zone": [0],
        "vector": [-1.0, 0.0, 0.0],
        "mach": mach,
    },
}
