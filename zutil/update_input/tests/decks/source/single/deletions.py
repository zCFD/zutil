parameters = {
    "restart": False,
    "units": "SI",
    "time marching": {"cycles": 10, "cfl": 1.0},
}
del parameters["units"]
parameters["time marching"].pop("cfl")
