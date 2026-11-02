def settle(**kwargs):
    t = kwargs["time"]
    if t > 1.0:
        return settle(**{"time": 1.0, "time step": kwargs["time step"]})
    return {"velocity": (t, 0.0, 0.0)}


parameters = {
    "FZ_1": {
        "type": "translating",
        "zone": [0],
        "vector": [1.0, 0.0, 0.0],
        "translation function": settle,
    },
}
