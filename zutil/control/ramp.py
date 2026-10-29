"""
Copyright (c) 2012-2026, Zenotech Ltd
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

from typing import Callable, Optional


def linear_ramp(
    increment: float,
    start_cycle: int = 1,
    end_cycle: Optional[int] = None,
    initial: Optional[float] = None,
) -> Callable[..., float]:
    """
    Creates a linear CFL ramp function.

    Args:
        increment: The amount to add to the CFL per solve cycle.
        start_cycle: The cycle at which to start incrementing (1-indexed).
        end_cycle: The cycle at which to stop incrementing.
        initial: CFL to start the ramp from. Returned at ``start_cycle``
            itself, so the ramp sets its own starting point rather than
            needing the solver to be told separately.

    Returns:
        A ramp function the solver calls as ``f(cycle=..., current_cfl=...)``.
    """

    def _ramp(*, cycle: int, current_cfl: float) -> float:
        if initial is not None and cycle == start_cycle:
            return initial
        if cycle < start_cycle or (end_cycle is not None and cycle > end_cycle):
            return current_cfl
        return current_cfl + increment

    return _ramp


def linear_stepped_ramp(
    increment: float,
    step_cycles: int,
    start_cycle: int = 1,
    end_cycle: Optional[int] = None,
    initial: Optional[float] = None,
) -> Callable[..., float]:
    """
    Creates a stepped linear CFL ramp function.

    Args:
        increment: The amount to add to the CFL at each step.
        step_cycles: The number of cycles between each step.
        start_cycle: The cycle at which to start incrementing (1-indexed).
        end_cycle: The cycle at which to stop incrementing.
        initial: CFL to start the ramp from, returned at ``start_cycle``.

    Returns:
        A ramp function the solver calls as ``f(cycle=..., current_cfl=...)``.
    """

    def _ramp(*, cycle: int, current_cfl: float) -> float:
        if initial is not None and cycle == start_cycle:
            return initial
        if cycle < start_cycle or (end_cycle is not None and cycle > end_cycle):
            return current_cfl
        if (cycle - start_cycle + 1) % step_cycles == 0:
            return current_cfl + increment
        return current_cfl

    return _ramp


def exponential_ramp(
    factor: float,
    start_cycle: int = 1,
    end_cycle: Optional[int] = None,
    max_allowed: float = 1e6,
    initial: Optional[float] = None,
) -> Callable[..., float]:
    """
    Creates an exponential CFL ramp function.

    Args:
        factor: The multiplication factor per cycle (> 1.0).
        start_cycle: The cycle at which to start incrementing (1-indexed).
        end_cycle: The cycle at which to stop incrementing.
        max_allowed: Guard protection against floating point overflow.
        initial: CFL to start the ramp from, returned at ``start_cycle``.

    Returns:
        A ramp function the solver calls as ``f(cycle=..., current_cfl=...)``.
    """
    if factor < 1.0:
        raise ValueError("Exponential ramp factor must be >= 1.0")

    def _ramp(*, cycle: int, current_cfl: float) -> float:
        if initial is not None and cycle == start_cycle:
            return initial
        # Skip start_cycle itself so the initial CFL is held for one cycle,
        # giving cfl = initial * factor^(cycle - start_cycle).
        if cycle <= start_cycle or (end_cycle is not None and cycle > end_cycle):
            return current_cfl
        try:
            nxt = current_cfl * factor
            return min(nxt, max_allowed)
        except OverflowError:
            return max_allowed

    return _ramp


def sequence_ramps(
    *ramps: Callable[..., float],
) -> Callable[..., float]:
    """
    Combines multiple ramp functions into a single sequenced ramp function.
    Functions that return modifications to `current_cfl` can be executed sequentially.

    Args:
        *ramps: An arbitrary number of callable ramp functions.

    Returns:
        A callable ramp function.
    """

    def _ramp(*, cycle: int, current_cfl: float) -> float:
        cfl = current_cfl
        for r in ramps:
            cfl = r(cycle=cycle, current_cfl=cfl)
        return cfl

    return _ramp
