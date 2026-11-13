def schedule(**kwargs):
    return {"velocity": (kwargs["time"], 0.0, 0.0)}


def rotor_motion(**kwargs):
    return schedule(**kwargs)


parameters = {
    "FZ_1": {
        "type": "translating",
        "zone": [0],
        "vector": [1.0, 0.0, 0.0],
        "translation function": rotor_motion,
    },
}
