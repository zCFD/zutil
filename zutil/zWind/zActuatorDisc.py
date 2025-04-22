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

Module to contain functionality of actuator disc methods
"""

from zutil.zWind.zTurbine import zTurbine
import numpy as np
from zutil import R_2vect
from typing import Optional
import matplotlib.pyplot as plt


class zActuatorDiscSegment:
    """
    Represents an individual actuator disc segment with associated attributes and methods.
    """

    def __init__(
        self,
        r: float,
        dr: float,
        theta: float,
        dtheta: float,
        disc_normal: np.ndarray,
        disc_centre: np.ndarray,
        rotation_direction: str = "clockwise",
    ) -> None:
        self.r = r
        self.dr = dr
        self.theta = theta
        self.dtheta = dtheta
        self.disc_normal = disc_normal
        self.disc_centre = disc_centre
        self.rotation_direction = rotation_direction

        self.rp = r + 0.5 * dr
        self.da = self.dtheta * self.rp * dr
        disc_theta = self.theta + 0.5 * self.dtheta

        # Calculate local_vector directly in global coordinates
        self.local_vector = (
            self.rp
            * np.array([np.cos(disc_theta), np.sin(disc_theta), 0.0])
            @ np.linalg.inv(self._rotation_matrix())
        )

        self.segment_centre = self.disc_centre + self.local_vector

    def _rotation_matrix(self) -> np.ndarray:
        """Constructs the rotation matrix from the disc normal."""
        return R_2vect(np.array([0, 0, 1]), self.disc_normal)

    def calculate_velocity_vector(self, ulocal: np.ndarray) -> None:
        """Calculates local velocity components for the segment."""
        rvec = self.local_vector / np.linalg.norm(self.local_vector)  # get unit vector

        if self.rotation_direction == "clockwise":
            local_omega_vec = np.cross(rvec, self.disc_normal)
        else:
            local_omega_vec = np.cross(self.disc_normal, rvec)

        self.v_n = -np.dot(ulocal, self.disc_normal)
        self.v_r = np.dot(ulocal, local_omega_vec)
        self.omega_air = np.dot(local_omega_vec, ulocal) / self.rp
        self.u_ref_local = np.dot(ulocal, self.disc_normal)


class zActuactorDisc:
    """Main class for handling actuator disc operations"""

    def __init__(self, control_dict: dict, verbose: bool = False) -> None:
        # create turbine model

        self.n_segments = control_dict["discretisation"]["number of elements"]
        self.verbose = verbose

        self.turbine = zTurbine(control_dict)

        self.create_annulus()

    def __enter__(self) -> object:
        return self

    def __exit__(self, type, value, traceback) -> bool:  # noqa: ANN001
        return False

    def create_annulus(self) -> list:
        """Discretise the area of the disc, and store in list of annulus segments"""
        self.annulus = []
        self.dtheta = np.deg2rad(360.0 / self.n_segments)
        for i in range(self.n_segments):
            r = self.turbine.inner_radius
            while r < self.turbine.outer_radius:
                dr = (
                    self.dtheta
                    * max(r, 0.01 * self.turbine.outer_radius)
                    / (1.0 - 0.5 * self.dtheta)
                )
                max_r = r + dr
                if max_r > self.turbine.outer_radius:
                    dr = self.turbine.outer_radius - r

                self.annulus.append(
                    zActuatorDiscSegment(
                        r,
                        dr,
                        i * self.dtheta,
                        self.dtheta,
                        self.turbine.normal,
                        self.turbine.centre,
                    )
                )

                r += dr

        return self.annulus

    def get_annulus(self) -> list:
        """Returns the annulus data in a C++-compatible format."""
        return [
            (
                seg.r,
                seg.dr,
                seg.theta,
                seg.dtheta,
                seg.segment_centre[0],
                seg.segment_centre[1],
                seg.segment_centre[2],
            )
            for seg in self.annulus
        ]

    def create_turbine_segments(
        self,
        u_ref: float,
        density: float,
        annulusVel: Optional[list] = None,
        turbine_name_dict: Optional[dict] = {},
        turbine_name: str = "",
    ) -> list:
        """Main body of actuator disc code- take in velocities at the turbine disc, return the torque and moment at each section"""
        if annulusVel:
            # if annulus_vec is supplied, u_ref = mean of annulus vec cells
            temp = np.reshape(annulusVel, (-1, 3)).T
            self.u_ref = np.sqrt(
                np.mean(temp[0]) ** 2 + np.mean(temp[1]) ** 2 + np.mean(temp[2]) ** 2
            )
        else:
            # take u_ref from code
            self.u_ref = u_ref

        self.density = density

        # reshape list of annular velocities into nx3 vector
        if annulusVel:
            self.annulus_velocities = np.reshape(annulusVel, (-1, 3))
        else:
            self.annulus_velocities = []

        converged = False

        # controller loop- get initial turbine conditions
        pitch, omega = self.turbine.controller.get_conditions()
        while not converged:
            # self.turbine.set_omega(omega)
            # self.turbine.set_pitch(pitch)
            # get performance for current operating conditions
            self.calculate_turbine_performance(
                pitch, turbine_name_dict=turbine_name_dict, turbine_name=turbine_name
            )

            # check for controller convergence
            pitch, omega, converged = self.turbine.controller.update_conditions(
                self.u_ref, self.torque
            )

        return self.annulus_out

    def calculate_turbine_performance(
        self,
        pitch: float,
        turbine_name_dict: Optional[dict] = None,
        turbine_name: str = "",
    ) -> None:
        # get predicted power and torque of turbine

        total_area = 0
        total_thrust = 0
        total_torque = 0

        self.annulus_out = []

        for i, seg in enumerate(self.annulus):
            if len(self.annulus_velocities) > 0:
                seg.calculate_velocity_vector(self.annulus_velocities[i])
            dt, dq = self.turbine.get_discrete_performance(self.u_ref, seg)
            self.annulus_out.append((dt, dq, seg.r, seg.dr, seg.theta, seg.dtheta))

            total_area += seg.da
            total_thrust += dt
            total_torque += np.abs(dq * seg.rp)

        total_power = total_torque * self.turbine.omega

        # Report back to turbine power
        turbine_name_dict[turbine_name + "_power"] = total_power
        turbine_name_dict[turbine_name + "_uref"] = self.u_ref
        turbine_name_dict[turbine_name + "_omega"] = self.turbine.omega
        turbine_name_dict[turbine_name + "_thrust"] = total_thrust

        self.torque = total_torque

    # Visualisation aids
    def plot_annulus(self) -> plt.Axes:
        """ "Plot a wireframe of the returned annulus"""
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # unpack annulus out vector
        nt = self.n_segments
        num_pts = len(self.annulus_out)
        nr = int(num_pts / nt)

        R = np.zeros((nr, nt))
        T = np.zeros((nr, nt))
        Tr = np.zeros((nr, nt))

        ii = 0

        for t in range(nt):
            for r in range(nr):
                R[r, t] = self.annulus_out[ii][2]
                T[r, t] = self.annulus_out[ii][4]
                Tr[r, t] = self.annulus_out[ii][1]

                ii += 1

        X, Y = R * np.cos(T), R * np.sin(T)

        # Stitch across last row
        X = np.column_stack((X, X[:, 0]))
        Y = np.column_stack((Y, Y[:, 0]))
        Tr = np.column_stack((Tr, Tr[:, 0]))

        ax.plot_wireframe(X, Y, Tr)

        return ax


def convolution(
    disc: list,
    disc_centre: tuple,
    disc_radius: float,
    disc_normal: type,
    disc_up: tuple,
    cell_centre_list: list,
    cell_volume_list: list,
) -> tuple:
    """
    Performs a convolution of discrete disc annulus data to then background CFD mesh.

    Args:
        disc: (list) Representation of the disc.
        disc_centre: (numpy.ndarray) Centre of the disc.
        disc_radius: (float) Radius of the disc.
        disc_normal: (numpy.ndarray) Normal vector of the disc.
        disc_up: (numpy.ndarray) Up vector of the disc.
        cell_centre_list: (list) List of cell centres.
        cell_volume_list: (list) List of cell volumes.

    Returns:
        list: List of scaled cell forces, where each element is a tuple of (x, y, z) forces.
    """

    from mpi4py import MPI
    import libconvolution as cv

    cell_centre_list_np = np.asarray(cell_centre_list)
    cell_volume_list_np = np.asarray(cell_volume_list)

    weighted_sum = np.zeros(len(disc))

    weighted_sum = cv.convolution_2dkernel_weights(
        disc,
        disc_centre,
        disc_radius,
        disc_normal,
        disc_up,
        cell_centre_list_np,
        cell_volume_list_np,
    )
    # Need to reduce weighted sum over all processes
    totals = np.zeros_like(weighted_sum)

    MPI.COMM_WORLD.Allreduce(weighted_sum, totals, op=MPI.SUM)
    weighted_sum = totals

    thrust_check_total = 0

    cell_force = np.zeros(len(cell_centre_list_np) * 3)
    thrust_check = cv.convolution_2dkernel_force(
        disc,
        disc_centre,
        disc_radius,
        disc_normal,
        disc_up,
        cell_centre_list_np,
        cell_volume_list_np,
        weighted_sum,
        cell_force,
    )

    thrust_check_array = np.array([thrust_check])
    thrust_check_total_array = np.array([0.0])

    MPI.COMM_WORLD.Allreduce(thrust_check_array, thrust_check_total_array, op=MPI.SUM)
    thrust_check_total = thrust_check_total_array[0]
    thrust_check = thrust_check_total

    # thrust_check = 0.0
    total_thrust = 0.0
    for idx, w in enumerate(weighted_sum):
        segment = disc[idx]
        # if w > 0.0:
        #    thrust_check += segment[0]
        total_thrust += segment[0]

    # Broken: Cell_force_scaled will have a different struct to cell_force
    if thrust_check > 0.0:
        thrust_factor = total_thrust / thrust_check
        # if MPI.COMM_WORLD.Get_rank() == 0:
        cell_force_scaled = []
        for cell in range(len(cell_force) // 3):
            cell_force_scaled.append(
                (
                    cell_force[cell * 3 + 0] * thrust_factor,
                    cell_force[cell * 3 + 1] * thrust_factor,
                    cell_force[cell * 3 + 2] * thrust_factor,
                )
            )
        return cell_force_scaled
    else:
        cell_force = np.ndarray.tolist(cell_force)
        cell_array = iter(cell_force)
        return list(zip(cell_array, cell_array, cell_array))

    cell_force = np.ndarray.tolist(cell_force)
    cell_array = iter(cell_force)
    return list(zip(cell_array, cell_array, cell_array))


def main(
    turbine_zone_dict: dict,
    u_ref: float,
    density: float,
    turbine_name_dict: Optional[dict] = {},
    turbine_name: Optional[str] = "",
    annulusVel: Optional[list] = None,
    annulusTi: Optional[list] = None,
) -> list:
    """
    Creates turbine segments using an actuator disc model.

    Args:
        turbine_zone_dict: Dictionary of turbine zone information.
        v0, v1, v2: Velocity components.
        density: Fluid density.
        turbine_name_dict: (optional) Dictionary of turbine names- required for report output.
        turbine_name: (optional) Name of the turbine- required for report output.
        annulusVel: (optional) Annulus velocity.
        annulusTi: (optional) Annulus turbulence intensity.

    Returns:
        Annulus data list.
    """

    ad = zActuactorDisc(turbine_zone_dict)  # Use more descriptive variable name
    annulus = ad.create_turbine_segments(
        u_ref,
        density,
        annulusVel=annulusVel,
        turbine_name=turbine_name,
        turbine_name_dict=turbine_name_dict,
    )
    return annulus


def create_annulus(turbine_zone_dict: dict) -> list:
    """
    Creates an annulus using an actuator disc model.

    Args:
        turbine_zone_dict: Dictionary of turbine zone information.

    Returns:
        Annulus data list.
    """

    with zActuactorDisc(turbine_zone_dict) as ad:  # Use context manager
        ad.create_annulus()
        annulus = ad.get_annulus()
    return annulus
