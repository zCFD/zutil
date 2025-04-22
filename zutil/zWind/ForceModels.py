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

Force models used in zWind
"""

import numpy as np
from zutil.zWind import zTurbine
from zutil.converters import radps2rps


class zTurbineBase:
    def __init__(self, model_dict: dict) -> None:
        """Base class for turbine and propellor modelling"""
        pass


class zSimpleFixed(zTurbineBase):
    def __init__(self, model_dict: dict) -> None:
        super().__init__(model_dict)

        self.power = None
        self.power_coefficient = None
        self.thrust = None
        self.thrust_coefficient = None

    def get_power(self, u_ref: float) -> float:
        return self.power

    def get_power_coefficient(self, u_ref: float) -> float:
        return self.power_coefficient

    def get_thrust(self, u_ref: float) -> float:
        return self.thrust

    def get_thrust_coefficient(self, u_ref: float) -> float:
        return self.thrust_coefficient

    def get_performance(
        u_ref: float, density: float, turbine: zTurbine
    ) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def discrete_performance(
        self, u_ref: float, density: float, seg: object, turbine: zTurbine
    ) -> tuple:
        thrust, power, torque = self.get_performance(u_ref, density, turbine)

        dt = thrust * (seg.da / turbine.rotor_swept_area)
        dq = torque * (seg.da / (turbine.rotor_swept_area * seg.rp))

        if turbine.rotation_direction == "anticlockwise":
            dq = -dq

        return (dt, dq)


class zTurbineSimpleFixed(zSimpleFixed):
    """Turbine with simple power model at fixed operating point- no controller, no curve"""

    def __init__(self, model_dict: dict) -> None:
        super().__init__(model_dict)
        self.model_type = "Turbine Simple Fixed"

        self.power = model_dict["power"]
        self.thrust_coefficient = model_dict["thrust coefficient"]

    def get_performance(self, u_ref: float, density: float, turbine: zTurbine) -> tuple:
        thrust_coefficient = self.get_thrust_coefficient(u_ref)
        power = self.get_power(u_ref)

        pitch, omega = turbine.controller.get_conditions()

        if omega != 0.0:
            torque = power / omega
            thrust = (
                0.5
                * density
                * u_ref**2
                * turbine.rotor_swept_area
                * thrust_coefficient
            )
        else:
            torque = 0.0
            thrust = 0.0
            power = 0.0

        return (thrust, power, torque)


class zPropellorSimpleFixed(zSimpleFixed):
    """Propellor with simple power model at fixed operating point- no controller, no curve"""

    def __init__(self, model_dict: dict) -> None:
        super().__init__(model_dict)
        self.model_type = "Propellor Simple Fixed"

        self.thrust_coefficient = model_dict["thrust coefficient"]
        self.power_coefficient = model_dict["power coefficient"]

    def get_performance(self, u_ref: float, density: float, turbine: zTurbine) -> tuple:
        ct = self.get_thrust_coefficient(u_ref)
        cp = self.get_power_coefficient(u_ref)

        pitch, omega = turbine.controller.get_conditions()

        if omega != 0.0:
            power = (
                cp * density * radps2rps(omega) ** 3 * (turbine.outer_radius * 2) ** 5
            )
            torque = power / omega
            thrust = ct * radps2rps(omega) ** 2 * (turbine.outer_radius * 2) ** 4
        else:
            thrust = 0.0
            power = 0.0
            torque = 0.0

        return (thrust, power, torque)


class zTurbineSimpleCurve(zTurbineSimpleFixed):
    """Turbine with simple power model with power curve"""

    def __init__(self, model_dict: dict) -> None:
        super().__init__(model_dict)
        self.model_type = "Turbine Simple Curve"

        self.cp_curve = np.array(model_dict["power curve"])
        self.ct_curve = np.array(model_dict["thrust coefficient curve"])

    def get_power(self, u_ref: float) -> float:
        return np.interp(
            u_ref, self.cp_curve[:, 0], self.cp_curve[:, 1], left=0, right=0
        )

    def get_thrust_coefficient(self, u_ref: float) -> float:
        return np.interp(
            u_ref, self.ct_curve[:, 0], self.ct_curve[:, 1], left=0, right=0
        )


class zTurbineInductionFixed(zTurbineBase):
    """"""

    def __init__(self, control_dict: dict) -> None:
        super().__init__(control_dict)
        self.model_type = "Turbine induction"

    def get_axial_induction_factor(self, ct: float) -> float:
        return (4.0 - np.sqrt(4.0 * 4.0 - 4.0 * 4.0 * ct)) / (2.0 * 4.0)

    def discrete_performance(self, u_ref: float, density: float, seg: object) -> tuple:
        ct = self.get_thrust_coefficient(u_ref)
        tsr = self.get_tsr(u_ref)

        ind_fac = self.get_axial_induction_factor(ct)

        omega = (tsr * u_ref) / self.outer_radius

        dt = 0.5 * density * u_ref**2 * seg.da * 4.0 * ind_fac * (1.0 - ind_fac)

        lambda_r = (seg.rp * omega) / u_ref
        if lambda_r > 0.0:
            angular_induction_factor = -0.5 + np.sqrt(
                0.25 + ((ind_fac * (1.0 - ind_fac)) / (lambda_r**2))
            )
        else:
            angular_induction_factor = 0.0

        dq = (
            4.0
            * angular_induction_factor
            * (1 - ind_fac)
            * 0.5
            * density
            * u_ref
            * omega
            * seg.rp
            * seg.rp
            * seg.da
            * seg.rp
        )

        if self.rotation_direction == "anticlockwise":
            dq = -dq

        return (dt, dq)
