def leaf_area(*, locations, wall_distances):
    return [0.1 for _ in locations]


parameters = {
    "FZ_1": {
        "type": "canopy",
        "def": "forest.vtp",
        "func": leaf_area,
        "cd": 0.2,
        "beta_p": 1.0,
        "beta_d": 4.0,
        "ceps_4": 0.9,
        "ceps_5": 0.8,
    },
}
