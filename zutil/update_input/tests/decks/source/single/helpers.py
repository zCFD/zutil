import math

import numpy as np
import zutil

alpha = 2.0

parameters = {
    "IC_1": {
        "temperature": zutil.to_kelvin(540.0),
        "V": {"vector": zutil.vector_from_angle(alpha, 0.0)},
        "speed of sound": math.sqrt(1.4 * 287.0 * 288.15),
    },
    "mesh transform matrix": np.eye(3).tolist(),
    "scale": np.array([1.0, 2.0, 3.0]),
}
