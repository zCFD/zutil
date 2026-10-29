import zutil
import numpy as np
import zutil.transform

# Set up an arbitray rotation of the domain to test numerics and limiter for co-ordinate invariance
rot_vec = [1.0, 2.0, 3.0]  # an arbitrary axis of rotation
rot_ang_deg = np.rad2deg(1.0)  # angle of rotation in radians

parameters = zutil.get_parameters_from_file(
    "shocktube_fv_second_tf_base.py", namespace=globals()
)

parameters["equations"] = "euler"
parameters["euler"] = {"order": "second", "entropy fix": 0.05}

parameters["time marching"] = {
    "unsteady": {"total time": 0.2, "time step": 1.0e-5},
    "scheme": {"name": "euler", "kind": "global timestepping"},
}
