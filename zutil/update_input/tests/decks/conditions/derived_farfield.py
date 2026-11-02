parameters = {
    "equations": "RANS",
    "reference": "IC_1",
    "IC_1": {
        "temperature": 288.15,
        "pressure": 101325.0,
        "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.2},
        "Reynolds No": 6.0e6,
    },
    "IC_2": {"reference": "IC_1", "total pressure ratio": 1.05},
    "BC_1": {"zone": [1], "type": "farfield", "kind": "riemann", "condition": "IC_2"},
}
