dt = 0.25

parameters = {
    "restart": False,
    "time marching": {"unsteady": {"total time": 1.0, "time step": dt}, "cycles": 10},
    "BC_1": {"zone": [1], "type": "wall"},
}
