"""
Copyright (c) 2012-2024, Zenotech Ltd
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Zenotech Ltd nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ZENOTECH LTD BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
from typing import Any
from mpi4py import MPI


# Optimal power coefficient as a function of (a function of) tip speed ratio
# "A Compact, Closed-form Solution for the Optimum, Ideal Wind Turbine" (Peters, 2012)
def read_default(dict: dict, key: str, default_val: Any, verbose: bool = True) -> Any:
    if key in dict:
        value = dict[key]
    else:
        value = default_val
        dict[key] = default_val
        if MPI.COMM_WORLD.Get_rank() == 0 and verbose:
            if "name" in dict:
                print(
                    "Turbine zone "
                    + str(dict["name"])
                    + " missing: "
                    + str(key)
                    + " - setting to "
                    + str(default_val)
                )
            else:
                print(
                    "Turbine zone missing name and missing: "
                    + str(key)
                    + " - setting to "
                    + str(default_val)
                )
    return value


def glauert_peters(y: float) -> float:
    """
    Calculates the Glauert-Peters correction factor for a given Mach number.

    Args:
        y: The Mach number (float).

    Returns:
        The Glauert-Peters correction factor (float).
    """

    denominator = 27.0 * (1.0 + y / 4.0)
    power_coeff = (
        16.0
        * (1.0 - 2.0 * y)
        / denominator
        * (
            1.0
            + (457.0 / 1280.0) * y
            + (51.0 / 640.0) * y**2
            + y**3 / 160.0
            + 3.0
            / 2.0
            * y
            * (np.log(2.0 * y) + (1.0 - 2.0 * y) + 0.5 * (1.0 - 2.0 * y) ** 2)
            / denominator**2
        )
    )
    return power_coeff
