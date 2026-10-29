def shock(**kwargs):
    return {"pressure": 1.0}


def ramp(cycle):
    return min(cycle * 0.1, 5.0)


parameters = {
    "initial": {"name": "IC_1", "func": shock},
    "time marching": {"cfl ramp": ramp, "cfl transport": lambda c: 2.0 * c},
}
