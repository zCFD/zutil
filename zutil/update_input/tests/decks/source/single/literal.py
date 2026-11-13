"""A deck written as one literal."""

parameters = {
    "restart": False,
    "time marching": {
        "unsteady": {"total time": 1.0, "time step": 1.0},
        "scheme": {"name": "runge kutta", "stage": 5},
        "cycles": 100,
    },
    "IC_1": {"temperature": 288.15, "V": {"vector": [1.0, 0.0, 0.0]}},
    "BC_1": {"zone": [1, 2], "type": "wall"},
    "write output": {"volume variables": ("V", "p"), "options": {}},
}

if __name__ == "__main__":
    print(parameters)
