import zutil

mach = 0.5
t_end = 1.0
cycles = 100
dt = 0.01

parameters = zutil.get_parameters_from_file("ns_base", namespace=globals())

parameters["time marching"] = {
    "unsteady": {"total time": t_end, "time step": dt},
    "scheme": {"kind": "global timestepping"},
}
parameters["BC_2"] = {"zone": [2], "type": "farfield"}
