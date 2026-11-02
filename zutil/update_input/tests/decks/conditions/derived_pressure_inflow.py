parameters = {
    "equations": "RANS",
    "reference": "IC_1",
    "IC_1": {
        "temperature": 288.15,
        "pressure": 101325.0,
        "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.2},
        "Reynolds No": 6.0e6,
    },
    "IC_2": {
        "reference": "IC_1",
        "total pressure ratio": 1.2,
        "total temperature ratio": 1.0,
    },
    "BC_4": {"zone": [2], "type": "inflow", "kind": "default", "condition": "IC_2"},
}
