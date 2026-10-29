def lurch(t, dt):
    return {"velocity": (t, 0.0, 0.0)}


parameters = {
    "FZ_1": {
        "type": "translating",
        "zone": [0],
        "vector": [1.0, 0.0, 0.0],
        "translation function": lurch,
    },
}
