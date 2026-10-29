import copy

dt = 0.5
base = {
    "time marching": {"unsteady": {"total time": 10.0, "time step": dt}},
    "restart": False,
}

parameters = copy.deepcopy(base)
parameters["restart"] = True
