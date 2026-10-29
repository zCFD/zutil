def linear_accel(**kwargs):
    return {"velocity": (-69.4 * min(kwargs["time"] / 1.5, 1.0), 0.0, 0.0)}


parameters = {
    "FZ_1": {
        "type": "translating",
        "zone": [0],
        "vector": [-69.4, 0.0, 0.0],
        "translation function": linear_accel,
    },
}
