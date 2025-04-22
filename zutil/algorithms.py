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

Useful algorithms
"""

from typing import Callable
import numpy as np


def gss(f: Callable, a: float, b: float, tol: float = 1e-5) -> tuple[float, float]:
    """
    Golden section search:

    Given a function `f` with a single local maximum in the interval `[a, b]`,
    returns an interval `[c, d]` containing the maximum with a width `d - c <= tol`.

    Args:
        f: The function to optimize (callable).
        a: The lower bound of the initial interval.
        b: The upper bound of the initial interval.
        tol: The desired tolerance (default: 1e-5).

    Returns:
        A tuple (c, d) representing the final interval containing the maximum.
    """

    golden_ratio = (np.sqrt(5) + 1) / 2  # phi
    golden_ratio_conjugate = 1 - golden_ratio  # 1 - phi

    (a, b) = (min(a, b), max(a, b))  # Ensure a <= b
    h = b - a

    if h <= tol:
        return (a, b)

    n = int(
        np.ceil(np.log(tol / h) / np.log(golden_ratio_conjugate))
    )  # Calculate iterations
    c = b - golden_ratio_conjugate * h  # Initial interior points
    d = a + golden_ratio_conjugate * h

    for _ in range(n - 1):
        if f(c) > f(d):
            b = d
            d = c
            h = golden_ratio_conjugate * h
            c = b - golden_ratio_conjugate * h
        else:
            a = c
            c = d
            h = golden_ratio_conjugate * h
            d = a + golden_ratio_conjugate * h

    return (a, d) if f(c) > f(d) else (c, b)  # Return final interval
