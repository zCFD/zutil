steady = True


def boundaries():
    return {"BC_1": {"zone": [1], "type": "wall"}}


if steady:
    marching = {"unsteady": {"total time": 1.0, "time step": 1.0}}
else:
    marching = {"unsteady": {"total time": 5.0, "time step": 0.1}}

parameters = {"time marching": marching, "restart": False}
parameters.update(boundaries())
for i, zone in enumerate([3, 4]):
    parameters["BC_%d" % (i + 2)] = {"zone": [zone], "type": "symmetry"}
