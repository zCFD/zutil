def gust(**kwargs):
    return {"pressure": kwargs["p"]}


parameters = {
    "equations": "euler",
    "reference": "IC_1",
    "initial": {"name": "IC_1", "func": gust},
    "IC_1": {
        "temperature": 1.0,
        "pressure": 1.0,
        "V": {"vector": [0.75, 0.0, 0.0]},
        "Reynolds No": 1.0,
    },
}
