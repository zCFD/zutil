def lift_target_ic(**kwargs):
    return {
        "temperature": 288.15,
        "pressure": 101325.0,
        "v": {"vector": [1.0, 0.0, 0.0], "mach": 0.2},
        "reynolds no": 6.0e6,
    }


parameters = {
    "equations": "RANS",
    "reference": "IC_1",
    "IC_1": {
        "temperature": 288.15,
        "pressure": 101325.0,
        "V": {"vector": [1.0, 0.0, 0.0], "Mach": 0.2},
        "Reynolds No": 6.0e6,
    },
    "IC_2": lift_target_ic,
    "BC_1": {"zone": [1], "type": "farfield", "kind": "riemann", "condition": "IC_2"},
}
