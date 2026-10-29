import zutil
import numpy as np


class MovingAverage(object):
    def __init__(self, N):
        self.N = N
        self.samples = np.zeros(self.N)
        self.num_samples = 0
        self.total = 0.0

    def add_sample(self, val):
        self.total += val
        if self.num_samples < self.N:
            self.samples[self.num_samples] = val
            self.num_samples += 1
        else:
            self.num_samples += 1
            oldest = self.samples[self.num_samples % self.N]
            self.total = self.total - oldest
            self.samples[self.num_samples % self.N] = val

    def moving_avg(self):
        return self.total / min(self.num_samples, self.N)


# https://scaron.info/blog/simple-linear-regression-with-online-updates.html
class SimpleLinearRegressor(object):
    def __init__(self):
        self.dots = np.zeros(5)
        self.intercept = None
        self.slope = 0.0

    def update(self, x: np.ndarray, y: np.ndarray):
        self.dots += np.array(
            [
                x.shape[0],
                x.sum(),
                y.sum(),
                np.dot(x, x),
                np.dot(x, y),
            ]
        )
        size, sum_x, sum_y, sum_xx, sum_xy = self.dots
        det = size * sum_xx - sum_x**2
        if det > 1e-10:  # determinant may be zero initially
            self.intercept = (sum_xx * sum_y - sum_xy * sum_x) / det
            self.slope = (sum_xy * size - sum_x * sum_y) / det


alpha_new = 0.0
d_cL_d_alpha = SimpleLinearRegressor()
num_updates = 0
cl_sample = [0.0, 0.0]
cl_moving_ave = MovingAverage(20)
dcl_ave = MovingAverage(10)


def my_transform(x, y, z):
    v = [x, y, z]
    v = zutil.rotate_vector(v, alpha_new, 0.0)
    return {"v1": v[0], "v2": v[1], "v3": v[2]}


def lift_target_ic_func(**kwargs):
    global alpha_new, cl_moving_ave, num_updates, cl_sample, d_cL_d_alpha
    alpha_init = 0  # Initial angle of attack
    alpha_init_inc = 0.5  # Initial alpha increment to compute dcl/dalpha
    lift_target = 0.10778e01  # Targeted lift coefficient - correct for alpha = 10
    update_period = 200  # How often alpha should be updated
    flow_settling_period = 500
    lift_break_tolerance = 1.0e-4  # Tolerance at which to stop the simulation
    terminate = False

    relaxation_factor = 0.75  # <1: under-relaxation, >1: over-relaxation

    update_required = False

    if "lift_target_alpha" in kwargs.keys():  # If timestep > 1
        alpha_current = kwargs["lift_target_alpha"]
        F_xyz = [kwargs["wall_Fx"], kwargs["wall_Fy"], kwargs["wall_Fz"]]
        F_LDS = zutil.rotate_vector(F_xyz, alpha_current, 0.0)
        cl_moving_ave.add_sample(
            F_LDS[2]
        )  # Lift coefficient, having rotated to account for alpha
        lift_current = cl_moving_ave.moving_avg()
    else:
        lift_current = 0  # placeholder value

    if kwargs["Cycle"] < flow_settling_period:
        alpha_new = alpha_init
    else:
        if (kwargs["Cycle"] % update_period) == 0:
            if num_updates == 0:
                cl_sample[0] = lift_current
                d_alpha = alpha_init_inc
            elif num_updates == 1:
                cl_sample[1] = lift_current
                # Update
                d_cL_d_alpha.update(
                    np.array([alpha_init, alpha_current]),
                    np.array([cl_sample[0], cl_sample[1]]),
                )
                d_cL_required = lift_target - cl_sample[1]
                dcl_ave.add_sample(d_cL_required)
                d_alpha = d_cL_required / d_cL_d_alpha.slope
            else:
                d_cL_required = lift_target - lift_current
                dcl_ave.add_sample(d_cL_required)
                # update
                d_cL_d_alpha.update(np.array([alpha_current]), np.array([lift_current]))

                d_alpha = relaxation_factor * d_cL_required / d_cL_d_alpha.slope

                if dcl_ave.moving_avg() < lift_break_tolerance:
                    terminate = True

            d_alpha = min(
                2.0, max(-2.0, d_alpha)
            )  # Not +-5 since that'd take us straight to the required alpha!

            alpha_new = alpha_current + d_alpha
            print(
                "Alpha updated from {} to {}".format(alpha_current, alpha_new),
                flush=True,
            )
            num_updates += 1
            if not terminate:
                update_required = True
        else:
            alpha_new = alpha_current

    dict_out = {
        "temperature": 300,
        "pressure": 101325.0,
        "v": {"mach": 0.15, "vector": zutil.vector_from_angle(alpha_new, 0.0)},
        "reynolds no": 6.0e6,
        "eddy viscosity ratio": 1.0,
        "monitor": {
            "prefix": "lift_target",
            "alpha": alpha_new,
            "cl": lift_current,
            "dclda": d_cL_d_alpha.slope,
            "update": update_required,
            "terminate": terminate,
        },
    }

    return dict_out


parameters = {
    "restart": False,
    # units for dimensional quantities
    "units": "SI",
    "safe": False,
    # reference state
    "reference": "IC_1",
    "time marching": {
        "unsteady": {"total time": 1.0, "time step": 1.0, "order": "second"},
        "scheme": {
            "name": "implicit euler",
            "stage": 1,
        },
        "cfl": 50,
        "cfl ramp factor": {"growth": 1.1, "initial": 1.0},
        "cycles": 5000,
    },
    "equations": "RANS",
    "RANS": {
        "order": "second",
        "limiter": "vanalbada",
        "inviscid flux scheme": "Roe",
        "precondition": True,
        "turbulence": {"model": "sst"},
    },
    "material": "air",
    "air": {
        "gamma": 1.4,
        "gas constant": 287.0,
        "Sutherlands const": 110.4,
        "Prandtl No": 0.72,
        "Turbulent Prandtl No": 0.9,
    },
    "IC_1": {
        "temperature": zutil.to_kelvin(540.0),
        "pressure": 101325.0,
        "V": {"vector": [0.0, 0.0, 0.0], "Mach": 0.15},
        #'viscosity' : 0.0,
        "Reynolds No": 6.0e6,
        "Reference Length": 1.0,
        "turbulence intensity": 5.2e-2,
        "eddy viscosity ratio": 1.0,
        "ambient turbulence intensity": 5.2e-2,
        "ambient eddy viscosity ratio": 1.0,
    },
    "IC_2": lift_target_ic_func,
    "BC_1": {
        "zone": [2, 3],
        "type": "symmetry",
    },
    "BC_2": {
        "zone": [4],
        "type": "wall",
        "kind": "noslip",
    },
    "BC_3": {
        "zone": [1],
        "type": "farfield",
        "condition": "IC_2",
        "kind": "riemann",
    },
    "report": {
        "frequency": 10,
        "forces": {
            "FR_1": {
                "name": "wall",
                "zone": [4],
                "transform": my_transform,
                "reference area": 1.0,
            }
        },
    },
    "write output": {
        "format": "vtk",
        "surface variables": [
            "V",
            "p",
            "T",
            "rho",
            "cp",
            "cf",
            "pressureforce",
            "pressuremoment",
            "frictionforce",
            "frictionmoment",
            "yplus",
        ],
        "volume variables": ["V", "p", "T", "rho", "mach", "cp", "eddy"],
    },
}
