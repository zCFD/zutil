coarse = True

if coarse:
    parameters = {"multigrid": 2, "restart": False}
else:
    parameters = {"multigrid": 4, "restart": False}

print("loaded")
