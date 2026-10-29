parameters = {
    "FZ_1": {
        "type": "disc",
        "name": "T01",
        "definition": "T01.vtp",
        "rotation direction": "clockwise",
        "discretisation": {"type": "disc", "number of elements": 24},
        "model": {"type": "simple", "kind": "turbine"},
        "controller": {"type": "none", "pitch": 2.0},
        "geometry": {
            "centre": [0.0, 0.0, 90.0],
            "normal": [1.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
        },
    },
}
