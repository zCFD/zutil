parameters = {"restart": False, "time marching": {"cycles": 10}}
parameters.update({"restart": True, "BC_1": {"zone": [1], "type": "wall"}})
parameters["time marching"].update({"cycles": 20, "cfl": 2.0})
parameters.update(partitioner="metis")
