def shock(**kwargs):
    return {"pressure": 0.1, "temperature": 0.8, "velocity": [0.0, 0.0, 0.0]}


parameters = {
    "equations": "euler",
    "reference": "IC_1",
    "initial": {"name": "IC_2", "func": shock},
    "IC_2": {
        "temperature": 0.8,
        "pressure": 0.1,
        "V": {"vector": [0.0, 0.0, 0.0]},
        "Reynolds No": 1.0,
    },
    "IC_1": {
        "temperature": 1.0,
        "pressure": 1.0,
        "V": {"vector": [0.75, 0.0, 0.0]},
        "Reynolds No": 1.0,
    },
}
