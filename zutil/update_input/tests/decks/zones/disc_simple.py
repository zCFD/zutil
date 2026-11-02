parameters = {
    "FZ_1": {
        "type": "disc",
        "name": "T01",
        "definition": "T01.vtp",
        "rotation direction": "clockwise",
        "reference plane": True,
        "reference point": [0.0, 0.0, 0.0],
        "verbose": True,
        "update frequency": 5,
        "discretisation": {"type": "disc", "number of elements": 24},
        "model": {
            "type": "simple",
            "kind": "turbine",
            "power model": "fixed",
            "power coefficient": 0.4,
            "thrust coefficient": 0.8,
        },
        "controller": {
            "type": "tsr curve",
            "pitch": 0.0,
            "tip speed ratio": 7.0,
            "tip speed ratio curve": [[5.0, 7.0], [10.0, 7.0]],
        },
        "geometry": {
            "centre": [0.0, 0.0, 90.0],
            "normal": [1.0, 0.0, 0.0],
            "up": [0.0, 0.0, 1.0],
            "inner radius": 2.0,
            "outer radius": 60.0,
        },
    },
}
