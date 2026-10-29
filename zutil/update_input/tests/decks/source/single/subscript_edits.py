dt = 0.1

parameters = {
    "time marching": {"unsteady": {"total time": 1.0, "time step": 1.0}},
    "BC_1": {"zone": [1], "type": "wall"},
}

parameters["time marching"]["unsteady"]["time step"] = dt
parameters["BC_2"] = {"zone": [2], "type": "farfield"}
parameters["time marching"]["cycles"] = 50
