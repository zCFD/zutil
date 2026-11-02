parameters = {
    "FZ_1": {
        "type": "disc",
        "name": "R01",
        "definition": "R01.vtp",
        "rotation direction": "clockwise",
        "discretisation": {"type": "disc", "number of elements": 24},
        "model": {
            "type": "BET",
            "kind": "propellor",
            "blade chord": [[0.2, 0.1], [1.0, 0.05]],
            "blade twist": [[0.2, 28.0], [1.0, 0.0]],
            "number of sections": 24,
            "number of blades": 2,
            "aerofoil positions": [[0.0, "Foil_A"], [1.0, "Foil_A"]],
            "aerofoils": {
                "Foil_A": {
                    "cd": [[-60.0, 1.4], [0.0, 0.03]],
                    "cl": [[-60.0, -0.5], [0.0, 0.2]],
                }
            },
            "tip loss correction": "rstar",
            "tip loss correction radius": 0.8,
        },
        "controller": {"type": "fixed", "omega": 314.0, "pitch": 0.0},
        "geometry": {
            "centre": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, -1.0],
            "up": [1.0, 0.0, 0.0],
            "inner radius": 0.04,
            "outer radius": 0.175,
        },
    },
}
