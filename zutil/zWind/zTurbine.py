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

Main class for turbine modelling functions
"""

import numpy as np
from zutil.zWind import ForceModels, zController, zBem


class zTurbine:
    def __init__(self, control_dict: dict) -> None:
        """Base class for turbine and propellor modelling"""
        self.name = control_dict["name"]

        # Geometry properties
        self.centre = control_dict["geometry"]["centre"]
        self.up = control_dict["geometry"]["up"]
        self.normal = control_dict["geometry"]["normal"]
        self.inner_radius = control_dict["geometry"]["inner radius"]
        self.outer_radius = control_dict["geometry"]["outer radius"]
        self.reference_plane = control_dict["reference plane"]

        # self.refrence_point = control_dict["reference point"]
        self.update_frequency = control_dict["update frequency"]

        # Model properties
        self.rotation_direction = control_dict.get("rotation direction", "clockwise")

        # Store remaining dictionaries incase they're needed later
        self.discretisation = control_dict["discretisation"]
        self.model = control_dict["model"]
        self.geometry = control_dict["geometry"]
        self.control = control_dict["controller"]
        self.spec = control_dict

        self.rotor_swept_area = np.pi * (
            self.outer_radius**2 - self.inner_radius**2
        )

        self.verbose = control_dict.get("verbose", False)

        self.assign_model()
        self.assign_controller()

    def assign_model(self) -> None:
        """Assign the physics model to be applied when calculating the force on the turbine

        The turbine can either be modelled using an aggregated thrust and torque curve, or modelled as individual blade elements
        """
        self.model_type = self.model["type"]

        # switch based on model type to load model
        if self.model_type == "simple":
            power_model = self.model["power model"]
            if power_model == "fixed":
                kind = self.model["kind"]
                if kind == "turbine":
                    self.turbine_model = ForceModels.zTurbineSimpleFixed(self.model)
                elif kind == "propellor":
                    self.turbine_model = ForceModels.zPropellorSimpleFixed(self.model)
            elif power_model == "curve":
                self.turbine_model = ForceModels.zTurbineSimpleCurve(self.model)

        elif self.model_type == "induction":
            power_model = self.model["power model"]
            if power_model == "fixed":
                self.turbine_model = ForceModels.zTurbineInductionFixed(self.model)
            elif power_model == "curve":
                self.turbine_model = ForceModels.zTurbineIndutionCurve(self.model)

        # BEMT options
        elif self.model_type == "BET":
            self.turbine_model = zBem.zTurbineBetFull(self.model)
        elif self.model_type == "BEMT":
            self.turbine_model = zBem.BEMT_Turbine(self.model)

        if self.verbose:
            print("Loaded: {}".format(self.turbine_model.model_type))

    def assign_controller(self) -> None:
        self.controller_type = self.control["type"]

        # switch based on controller type to load controller
        if self.controller_type == "fixed":
            self.controller = zController.FixedController(self.control)
        elif self.controller_type == "tsr curve":
            self.controller = zController.TSRController(self.control)
        elif self.controller_type == "schedule":
            self.controller = zController.zControllerSchedule(self.control)
        else:
            self.controller = zController.ControllerBase(self.control)

        if self.verbose:
            print("Loaded: {}".format(self.controller.controller_type))

    def get_performance(self, u_ref: float) -> None:
        """Get the total power and performance of the whole turbine"""
        if self.controller.iterative:
            pass
        else:
            self.pitch, self.omega = self.controller.set_conditions(u_ref, self)

        thrust, power, torque = self.turbine.get_performance(u_ref, self, 1.225)

        self.discretisation.discretise_performance(thrust, torque)

    def get_discrete_performance(self, u_ref: float, seg: object) -> None:
        """get the performance of a discretised section of a turbine disc- could be driven by actuator disc or line model"""

        # controller step- figure out operating conditions of turbine for given inflow conditions
        if self.controller.iterative:
            pass
        else:
            self.pitch, self.omega = self.controller.set_conditions(u_ref, self)

        # simulation step- use the force model to predict the power on the models
        dt, dq = self.turbine_model.discrete_performance(u_ref, 1.225, seg, self)
        return dt, dq
