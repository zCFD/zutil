def linear_accel(**kwargs):
    t = kwargs["time"]
    dt = kwargs.get("time step", 0.0)
    u = [-69.4, 0.0, 0.0]
    if t < 1.5:
        u[0] = -(t / 1.5) * 69.4
    return {"velocity": tuple(u)}


parameters = {
    "FZ_1": {
        "type": "translating",
        "zone": [0],
        "vector": [-1.0, 0.0, 0.0],
        "translation function": linear_accel,
    },
}
