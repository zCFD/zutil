parameters = {
    "equations": "RANS",
    "reference": "IC_1",
    "IC_1": {
        "temperature": 288.15,
        "pressure": 101325.0,
        "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.2},
        "Reynolds No": 6.0e6,
    },
    "IC_2": {"reference": "IC_1", "static pressure ratio": 0.9},
    "BC_5": {"zone": [3], "type": "outflow", "kind": "pressure", "condition": "IC_2"},
    "BC_6": {"zone": [4], "type": "outflow", "kind": "pressure", "condition": "IC_2"},
}
