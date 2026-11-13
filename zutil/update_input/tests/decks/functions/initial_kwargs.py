def shock(**kwargs):
    location = kwargs["location"]
    pressure = 0.1 if location[0] > 0.5 else 1.0
    return {"pressure": pressure, "temperature": 1.0, "velocity": [0.0, 0.0, 0.0]}


parameters = {
    "equations": "euler",
    "reference": "IC_1",
    "initial": {"name": "IC_1", "func": shock},
    "IC_1": {
        "temperature": 1.0,
        "pressure": 1.0,
        "V": {"vector": [0.75, 0.0, 0.0]},
        "Reynolds No": 1.0,
    },
}
