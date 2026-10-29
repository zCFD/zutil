def deform(*, location):
    return location


parameters = {
    "TR_1": {
        "type": "idw",
        "zone": [5],
        "func": deform,
        "alpha": 2.0,
        "power": [3.0, 5.0, 10.0],
        "fixed zones": [1, 2],
        "n nearest": 50,
        "deformation distance": 4.0,
        "blending stiffness": 0.5,
    },
}
