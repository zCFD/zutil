speed = mach * 340.0

parameters = {
    "time marching": {
        "unsteady": {"total time": t_end, "time step": t_end},
        "cycles": cycles,
    },
    "IC_1": {"V": {"vector": [speed, 0.0, 0.0]}},
    "BC_1": {"zone": [1], "type": "wall"},
}
