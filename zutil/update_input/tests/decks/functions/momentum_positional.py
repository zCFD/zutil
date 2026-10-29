def drive(cell_centres):
    return [(1.0, 0.0, 0.0) for _ in cell_centres]


parameters = {
    "FZ_1": {"type": "momentum", "def": "region.vtp", "func": drive},
}
