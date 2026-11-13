parameters = {
    "equations": "RANS",
    "reference": "IC_1",
    "restart": True,
    "restart casename": "wing_old",
    "restart ignore history": True,
    "IC_1": {
        "temperature": 288.15,
        "pressure": 101325.0,
        "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.2},
        "Reynolds No": 6.0e6,
    },
}
