dt = 1.0e-3
c_ref = 2.0
levels = 3

parameters = {
    "time marching": {"unsteady": {"total time": 100 * dt, "time step": dt}},
    "IC_1": {"reference length": 0.5 * c_ref, "V": {"vector": [c_ref, 0.0, 0.0]}},
    "multigrid": levels,
}
