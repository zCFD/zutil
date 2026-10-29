parameters = {
    "equations": "RANS",
    "reference": "IC_1",
    "IC_1": {
        "temperature": 288.15,
        "pressure": 101325.0,
        "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.2},
        "Reynolds No": 6.0e6,
    },
    "IC_3": {
        "reference": "IC_1",
        "mass flow rate": 2.5,
        "total temperature ratio": 1.0,
    },
    "bc_2": {"zone": [2], "type": "inflow", "kind": "massflow", "condition": "IC_3"},
}
