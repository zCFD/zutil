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

Placeholder module for controlling turbines in zCFD
"""

import numpy as np
from zutil import rpm2radps
from typing import Optional


class ControllerBase:
    """Base class for controller, equivalent to no controller"""

    def __init__(self, controller_dict: dict) -> None:
        self.pitch = 0.0
        self.omega = 0.0
        self.iterative = False
        self.controller_type = "Base"
        self.u = 0.0
        self.torque = 0.0

    def get_control_schedule(self) -> tuple:
        return 0.0, 0.0

    def check_convergence(self) -> bool:
        return True

    def get_conditions(self) -> tuple:
        return self.pitch, self.omega

    def update_conditions(
        self, u: Optional[float] = None, torque: Optional[float] = None
    ) -> tuple:
        if u:
            self.u = u

        if torque:
            self.torque = torque

        return (self.pitch, self.omega, True)

    def set_conditions(self, u_ref: float, torque: float) -> tuple:
        return (self.pitch, self.omega)


class FixedController(ControllerBase):
    def __init__(self, controller_dict: dict) -> None:
        super().__init__(controller_dict)

        self.iterative = False

        self.pitch = controller_dict["pitch"]
        self.omega = controller_dict["omega"]

        self.controller_type = "Fixed"

    def update_conditions(self, u_ref: float, torque: float) -> tuple:
        converged = True
        return self.pitch, self.omega, converged


class TSRController(ControllerBase):
    """No pitch control, derive omega from TSR curve"""

    def __init__(self, controller_dict: dict) -> None:
        super().__init__(controller_dict)

        self.iterative = False

        self.tsr_curve = np.array(controller_dict["tip speed ratio curve"])

        self.controller_type = "TSR"

    def get_tsr(self, u_ref: float) -> float:
        return np.interp(
            u_ref, self.tsr_curve[:, 0], self.tsr_curve[:, 1], left=0, right=0
        )

    def set_conditions(self, u_ref: float, turbine: object) -> tuple:
        tsr = self.get_tsr(u_ref)

        self.omega = (tsr * u_ref) / turbine.outer_radius
        return (self.pitch, self.omega)


class zControllerDynamic(ControllerBase):
    """zenotech dynamic turbine controller- will adjust the pitch of the turbine until the rated power is achieved, then will feather the turbine"""

    def __init__(self, control_dict: dict) -> None:
        super().__init__(control_dict)

        # turbine properties
        self.outer_radius = control_dict["outer radius"]
        self.rated_power = control_dict["rated power"]

        controller_dict = control_dict["controller"]

        # controller properties
        self.cut_in_speed = controller_dict["cut in speed"]
        self.cut_out_speed = controller_dict["cut out speed"]

        self.rated_power = controller_dict["rated power"]
        self.tip_speed_limit = controller_dict["tip speed limit"]

        self.friction_loss = controller_dict["friction loss"]

        self.blade_pitch_step = controller_dict["blade pitch step"]
        self.blade_pitch_range = controller_dict["blade pitch range"]
        self.blade_pitch_tol = controller_dict["blade pitch tol"]

        # initial conditions
        self.pitch = controller_dict["intial pitch"]
        self.omega = controller_dict["initial omega"]

        # conditions from previous step
        self.pitch_old = self.pitch
        self.omega_old = self.omega

        self.controller_type = "zController"

    def update_conditions(self, u_ref: float, torque: float) -> tuple:
        """takes an argument of the current operating conditions of the turbine, and returns the rotor velocity and blade pitch angle

        Right now will instantaneously update rather than allowing for rotor acceleration
        """

        # check any catastrophic cases
        if (u_ref < self.cut_int_speed) or (u_ref > self.cut_out_speed):
            # reference condition is outside of turbine operating range
            self.omega = 0.0

        # check turbine does not exceed tip speed limit
        # omega = min(omega, self.tip_speed_limit / self.outer_radius) * (
        #     1.0 - self.friction_loss
        # )

        # provided a disaster is avoided, decide on next values for pitch and turbine speed

        # determine point on operating curve
        if self.omega * torque >= self.rated_power:
            # feather turbine pitch
            pass
        elif self.omega * torque < self.rated_power:
            # pitch to maximise power
            pass

        # check if blade pitch is converged
        if abs(self.pitch - self.pitch_old) < self.blade_pitch_tol:
            converged = True

        return (self.pitch, self.omega, converged)


class zControllerSchedule(ControllerBase):
    """zenotech schedule turbine controller- will adjust the turbine pitch and speed to a specified control schedule given by a set of control curves- assumes you have run ROSCO or similar controller first"""

    def __init__(self, controller_dict: dict) -> None:
        super().__init__(controller_dict)
        self.pitch_schedule = np.array(controller_dict["pitch schedule"])
        self.speed_schedule = np.array(controller_dict["speed schedule"])

        self.cut_in_speed = self.pitch_schedule[0, 0]
        self.cut_out_speed = self.pitch_schedule[-1, 0]

        # convert omega from rpm to rad/s
        for i in range(self.speed_schedule.shape[0]):
            self.speed_schedule[i, 1] = rpm2radps(self.speed_schedule[i, 1])

        self.iterative = False

        self.controller_type = "Schedule"

    def get_init_conditions(self, u_ref: float) -> tuple:
        self.pitch = np.interp(
            u_ref, self.pitch_schedule[:, 0], self.pitch_schedule[:, 1]
        )
        self.omega = np.interp(
            u_ref, self.speed_schedule[:, 0], self.speed_schedule[:, 1]
        )

        return (self.pitch, self.omega)

    def set_conditions(self, u_ref: float, turbine: object) -> tuple:
        """lookup u_ref against control schedule and return specified pitch angle and turbine speed"""

        # check for out of range conditions
        if (u_ref < self.cut_in_speed) or (u_ref > self.cut_out_speed):
            self.omega = 0
            self.pitch = 0
            converged = True
            return self.pitch, self.omega

        self.pitch = np.interp(
            u_ref, self.pitch_schedule[:, 0], self.pitch_schedule[:, 1]
        )
        self.omega = np.interp(
            u_ref, self.speed_schedule[:, 0], self.speed_schedule[:, 1]
        )

        converged = True

        return (self.pitch, self.omega, converged)
