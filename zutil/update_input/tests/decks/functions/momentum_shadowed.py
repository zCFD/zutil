def drive(cells):
    scale = lambda cells: len(cells)
    return [(1.0 / scale(cells), 0.0, 0.0) for _ in cells]


parameters = {
    "FZ_1": {"type": "momentum", "def": "region.vtp", "func": drive},
}
