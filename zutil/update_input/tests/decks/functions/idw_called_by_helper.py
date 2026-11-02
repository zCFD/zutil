def rotate(x, y, z):
    return (-y, x, z)


def rotate_twice(x, y, z):
    return rotate(*rotate(x, y, z))


parameters = {
    "TR_1": {"type": "idw", "zone": [5], "func": rotate, "alpha": 2.0},
    "TR_2": {"type": "idw", "zone": [6], "func": rotate_twice, "alpha": 2.0},
}
