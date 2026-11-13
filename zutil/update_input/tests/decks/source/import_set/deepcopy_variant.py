import copy

import spread_base

parameters = copy.deepcopy(spread_base.parameters)
parameters["time marching"] = {"unsteady": {"total time": 2.0, "time step": 0.5}}
