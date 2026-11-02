time_step = 0.5


def ramp(**kwargs):
    return {"velocity": (kwargs["time step"] * time_step, 0.0, 0.0)}


parameters = {
    "FZ_1": {
        "type": "translating",
        "zone": [0],
        "vector": [1.0, 0.0, 0.0],
        "translation function": ramp,
    },
}
