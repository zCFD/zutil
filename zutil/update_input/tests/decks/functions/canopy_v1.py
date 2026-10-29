def leaf_area(cells):
    return [0.1 for _ in cells]


parameters = {
    "FZ_1": {"type": "canopy", "def": "forest.vtp", "func": leaf_area},
}
