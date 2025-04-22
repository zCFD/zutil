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

zCFD BEM implementation
"""

import numpy as np
from zutil.zWind import transforms, read_default
from typing import Optional


class zTurbineBetBase:
    """subclass for blade element propellor model"""

    def __init__(self, model_dict: dict) -> None:
        self.blade_twist = np.array(model_dict["blade twist"])
        self.blade_chord = np.array(model_dict["blade chord"])
        aerofoils = model_dict["aerofoil positions"]

        self.aerofoil_positions = []
        self.aerofoil_names = []

        for a in aerofoils:
            self.aerofoil_positions.append(a[0])
            self.aerofoil_names.append(a[1])

        self.aerofoils = model_dict["aerofoils"]

        self.tip_loss_correction = model_dict["tip loss correction"]
        self.tip_loss_correction_r = model_dict["tip loss correction radius"]

        self.n_blades = model_dict["number of blades"]

        self.kind = model_dict["kind"]

        self.model_type = "Bet base"

        # if self.tip_loss_correction:
        #     self.tip_loss_correction_r = model_dict["tip loss correction radius"]

    def interp_blade_twist(self, val: float) -> float:
        return np.interp(val, self.blade_twist[:, 0], self.blade_twist[:, 1])

    def interp_blade_chord(self, val: float) -> float:
        return np.interp(val, self.blade_chord[:, 0], self.blade_chord[:, 1])

    def interp_aerofoil_cl(self, val: float, pos: float) -> float:
        bindex = np.digitize(pos, self.aerofoil_positions, right=False)
        aerofoil = self.aerofoil_names[bindex]

        return np.interp(
            val,
            np.array(self.aerofoils[aerofoil]["cl"])[:, 0],
            np.array(self.aerofoils[aerofoil]["cl"])[:, 1],
        )

    def interp_aerofoil_cd(self, val: float, pos: float) -> float:
        bindex = np.digitize(pos, self.aerofoil_positions, right=False)
        aerofoil = self.aerofoil_names[bindex]

        return np.interp(
            val,
            np.array(self.aerofoils[aerofoil]["cd"])[:, 0],
            np.array(self.aerofoils[aerofoil]["cd"])[:, 1],
        )

    def get_tip_loss_factor(self, radial_position: float, turbine: object) -> float:
        if self.tip_loss_correction == "elliptic":
            tlc = np.sqrt(1.0 - (radial_position) ** 2)
        elif self.tip_loss_correction == "acos-fit":
            tlc = (2.0 / np.pi) * np.arccos(
                np.exp(-63.0 * (1.0 - (radial_position) ** 2.0))
            )
        elif self.tip_loss_correction == "acos shift-fit":
            tlc = (2.0 / np.pi) * np.arccos(
                np.exp(-48.0 * (1.0 - (radial_position) ** 2.0) - 0.5)
            )
        elif self.tip_loss_correction == "f-fit":
            tlc = (
                1.0
                - 2.5
                * ((1.0 - (radial_position) ** 2) ** 0.39)
                / (2.0 - (radial_position) ** 2) ** 64
            )
        elif self.tip_loss_correction == "rstar":
            if radial_position > self.tip_loss_correction_r:
                tlc = np.sqrt(
                    1.0
                    - (
                        (radial_position - self.tip_loss_correction_r)
                        / (1 - self.tip_loss_correction_r)
                    )
                    ** 2.0
                )
            else:
                tlc = 1.0
            return tlc
        else:
            return 1.0


class zTurbineBetProp(zTurbineBetBase):
    def __init__(self, control_dict: dict) -> None:
        super().__init__(control_dict)

    def get_sectional_AD_performance(
        self, u_ref: float, density: float, seg: object
    ) -> tuple[float, float]:
        # calculate local velocity vector
        if abs((seg.rp * self.omega) - seg.v_r) > 0.0:
            theta_rel = np.arctan(seg.v_n / ((seg.rp * self.omega) - seg.v_r))
        else:
            theta_rel = np.pi / 2.0

        u_rel = np.sqrt((seg.rp * self.omega - seg.v_r) ** 2 + seg.v_n**2)
        beta_twist = self.interp_blade_twist(seg.rp / self.outer_radius)
        chord = self.interp_blade_chord(seg.rp / self.outer_radius)

        beta = np.deg2rad(beta_twist)
        alpha = beta - theta_rel

        cl = self.interp_aerofoil_cl(alpha, seg.rp / self.outer_radius)
        cd = self.interp_aerofoil_cd(alpha, seg.rp / self.outer_radius)

        if self.tip_loss_correction:
            tlc = self.get_tip_loss_factor(seg)
            cl = cl * tlc
            cd = cd * tlc

        f_L = cl * 0.5 * density * u_rel**2 * chord
        f_D = cd * 0.5 * density * u_rel**2 * chord

        F_L = (self.n_blades / (2.0 * np.pi * seg.rp)) * f_L
        F_D = (self.n_blades / (2.0 * np.pi * seg.rp)) * f_D

        dt = -(F_L * np.cos(theta_rel) - F_D * np.sin(theta_rel)) * seg.da
        dq = -(F_L * np.sin(theta_rel) - F_D * np.cos(theta_rel)) * seg.da

        return dt, dq


class zTurbineBetFull(zTurbineBetBase):
    def __init__(self, model_dict: dict) -> None:
        super().__init__(model_dict)

        self.n_sections = model_dict["number of sections"]

        self.sections = np.linspace(0, 1, self.n_sections)

    def get_performance(self, u_ref: float, turbine: object, density: float) -> None:
        """Main body of BEMT code- work along each blade section, calculate the induced velocity vector at each section, then calculate sectional forces from polars"""

        # iteratate along span of blade
        for s in self.sections:
            # at each section caculate the induced velocity vector
            pass

    def discrete_performance(
        self, u_ref: float, density: float, seg: object, turbine: object
    ) -> tuple[float, float]:
        """takes an input of the desired rotor pitch angle, and returns the predicted performance of the turbine"""
        omega_rel = turbine.controller.omega - seg.omega_air
        u_rel = np.sqrt((seg.rp * omega_rel) ** 2 + seg.u_ref_local**2)

        if (seg.rp * omega_rel) > 0.0:
            theta_rel = np.arctan(seg.u_ref_local / (seg.rp * omega_rel))
        else:
            theta_rel = np.pi / 2.0

        radial_position = seg.rp / turbine.outer_radius

        beta_twist = self.interp_blade_twist(radial_position)
        chord = self.interp_blade_chord(radial_position) * turbine.outer_radius

        if self.kind == "propellor":
            theta_rel *= -1  # invert for sign convention

        beta = np.deg2rad(turbine.controller.pitch + beta_twist)
        alpha = theta_rel - beta

        if self.kind == "propellor":
            alpha *= -1  # invert for debugging

        cl = self.interp_aerofoil_cl(np.rad2deg(alpha), radial_position)

        tlc = self.get_tip_loss_factor(radial_position, turbine)
        cl *= tlc

        cd = self.interp_aerofoil_cd(np.rad2deg(alpha), radial_position)
        cd *= tlc

        f_L = cl * 0.5 * density * u_rel**2 * chord
        f_D = cd * 0.5 * density * u_rel**2 * chord

        F_L = (self.n_blades / (2.0 * np.pi * seg.rp)) * f_L
        F_D = (self.n_blades / (2.0 * np.pi * seg.rp)) * f_D

        dt = (F_L * np.cos(theta_rel) - F_D * np.sin(theta_rel)) * seg.da
        dq = (F_L * np.sin(theta_rel) + F_D * np.cos(theta_rel)) * seg.da

        if self.kind == "propellor":
            dq *= -1
            dt *= -1
        if turbine.rotation_direction == "anticlockwise":
            dq *= -1

        return dt, dq

    def init_bet_kernel(self, u_ref: float, density: float, seglist: list) -> None:
        pass

    def bet_kernel(self, pitch: float) -> None:
        pass


class Blade_Element:
    def __init__(
        self,
        p1: np.array,
        p2: np.array,
        chord: float,
        twist: float,
        aerofoil: str,
        verbose: bool = False,
    ) -> None:
        self.verbose = verbose
        # points defining each end of element
        self.p1 = p1
        self.p2 = p2

        # vector from element centre to centre of rotation
        self.r0 = (p2 + p1) / 2

        # aerodynamic properties
        self.chord = chord
        self.aerofoil = aerofoil

        # angular properties of each element
        self.twist = twist

        p12 = self.p2 - self.p1

        # figure these two out geometrically
        self.sweep = np.arctan(p12[1] / p12[0])
        self.prebend = -np.arctan(p12[1] / (np.sqrt(p12[0] ** 2 + p12[1] ** 2)))

        self.calculate_induced_angular_velocity()

        self.T_bc = transforms.T_bc(
            np.deg2rad(self.twist), np.deg2rad(self.sweep), np.deg2rad(self.prebend)
        )

        # figure out triad for specific element

    def calculate_induced_angular_velocity(self) -> None:
        omega = np.array([0.0, 0.79, 0.0])
        self.phi_induced = -np.cross(omega, self.r0)

    def global2element(self, V: np.array) -> None:
        return


class BEMT_Blade:
    def __init__(self, model_dict: dict, azimuth: float, verbose: bool = False) -> None:
        self.verbose = verbose

        # angular properties of each blade
        self.cone = read_default(model_dict, "cone", 0.0)
        self.pitch = 0.0
        self.num_elements = model_dict["number of elements"]
        self.azimuth = azimuth

        self.blade_axis = np.array(model_dict["blade axis"])
        self.blade_axis[:, 3] += 3.0
        # swap x and z to coincide with atom axis (FOR NOW)

        self.blade_axis[:, [1, 3, 2]] = self.blade_axis[:, [3, 2, 1]]

        self.chord = np.array(model_dict["blade chord"])
        self.twist = np.array(model_dict["blade twist"])

        aerofoils = model_dict["aerofoil positions"]

        self.aerofoil_positions = []
        self.aerofoil_names = []

        for a in aerofoils:
            self.aerofoil_positions.append(a[0])
            self.aerofoil_names.append(a[1])

        self.T_hb = transforms.T_hb(
            np.deg2rad(self.azimuth), np.deg2rad(self.cone), np.deg2rad(self.pitch)
        )

        # create elemments
        if self.verbose:
            print("discretising blade with {} elements".format(self.num_elements))
        self.elements = []
        self.nodes = np.linspace(0, 1.0, self.num_elements + 1)
        for i in range(self.num_elements):
            # points at either end of element
            p1 = self.get_blade_axis_point(self.nodes[i])
            p2 = self.get_blade_axis_point(self.nodes[i + 1])

            # chord, twist, aerofoil at midpoint
            chord = self.interp_blade_chord((self.nodes[i] + self.nodes[i + 1]) / 2)
            twist = self.interp_blade_twist((self.nodes[i] + self.nodes[i + 1]) / 2)
            aerofoil = self.get_aerofoil((self.nodes[i] + self.nodes[i + 1]) / 2)
            self.elements.append(Blade_Element(p1, p2, chord, twist, aerofoil))

    def get_blade_axis_point(self, radial_position: float) -> list:
        # returns the interpolated point on the blade axis at point `radial_position`
        point = np.array([0.0, 0.0, 0.0])
        for i, _ in enumerate(point):
            point[i] = np.interp(
                radial_position, self.blade_axis[:, 0], self.blade_axis[:, i + 1]
            )
        return point

    def interp_blade_chord(self, val: float) -> float:
        return np.interp(val, self.chord[:, 0], self.chord[:, 1])

    def interp_blade_twist(self, val: float) -> float:
        return np.interp(val, self.twist[:, 0], self.twist[:, 1])

    def get_aerofoil(self, pos: float) -> str:
        bindex = np.digitize(pos, self.aerofoil_positions, right=False)
        aerofoil = self.aerofoil_names[bindex]
        return aerofoil

    def create_elements(self) -> None:
        pass

    def get_induced_velocities(self) -> np.array:
        induced_velocities = np.zeros((self.num_elements, 3))
        for i, e in enumerate(self.elements):
            induced_velocities[i, :] = e.phi_induced

        return induced_velocities

    def blade2hub(self, V: np.array) -> np.array:
        # transform points in the blade coordinate system to the hub coordinate system
        return transforms.R_yy(np.deg2rad(self.azimuth)) @ V

    def blade2global(self, V: np.array) -> None:
        return


class BEMT_Turbine:
    def __init__(self, model_dict: dict, verbose: bool = False) -> None:
        self.verbose = verbose

        self.num_blades = model_dict["number of blades"]
        self.yaw = read_default(model_dict, "yaw", 0.0)
        self.tilt = read_default(model_dict, "tilt", 0.0)
        self.azimuth_offset = read_default(model_dict, "azimuth", 0.0)

        self.T_gh = transforms.T_gh(
            np.deg2rad(self.yaw), np.deg2rad(self.tilt), np.deg2rad(self.azimuth_offset)
        )

        # create blades
        self.blades = []
        if self.verbose:
            print("Creating {} blades".format(self.num_blades))
        azimuth_positions = (
            np.linspace(0, 360, self.num_blades + 1) + self.azimuth_offset
        )
        for i in range(self.num_blades):
            self.blades.append(BEMT_Blade(model_dict, azimuth_positions[i]))

        self.model_type = "BEMT Turbine"

    def discrete_performance(
        self, u_ref: float, density: float, seg: object, turbine: object
    ) -> tuple:
        return 0.0, 0.0

    def plot_turbine(self) -> None:
        # util for visualising the turbine

        nodes = []
        elements = []
        rp = []
        mechanical_velocities = []
        local_velocities = []
        i = 0
        for blade in self.blades:
            for e in blade.elements:
                elements.append([i, i + 1])

                # coordinates in blade axis
                triad = e.T_bc @ blade.T_hb @ self.T_gh
                p = np.linalg.inv(triad) @ e.p1
                nodes.append(p)

                r = np.linalg.inv(triad) @ e.r0
                rp.append(r)

                mechanical_velocity = np.linalg.inv(triad) @ e.phi_induced
                mechanical_velocities.append(mechanical_velocity)

                local_velocity = mechanical_velocity + np.array([0.0, 10.59, 0.0])
                local_velocities.append(local_velocity)

                i += 1
            nodes.append(np.linalg.inv(triad) @ e.p2)
            i += 1

        cell_types = np.array([3 for i in range(len(elements))])

        point_data = {}

        point_data["scalars"] = {}
        point_data["scalars"]["id"] = [i for i in range(len(nodes))]

        point_data["vectors"] = {}
        point_data["vectors"]["normal"] = [
            [float(i), float(i), float(i)] for i in range(len(nodes))
        ]

        element_data = {}
        element_data["scalars"] = {}
        element_data["scalars"]["id"] = [i for i in range(len(elements))]

        element_data["vectors"] = {}
        element_data["vectors"]["normal"] = [[i, i, i] for i in range(len(elements))]
        element_data["vectors"]["rp"] = rp
        element_data["vectors"]["mechanical_velocity"] = mechanical_velocities
        element_data["vectors"]["local_velocity"] = local_velocities

        write_vtk_fe_mesh(
            nodes,
            elements,
            cell_types,
            "test.vtk",
            point_data=point_data,
            element_data=element_data,
        )


def write_vtk_fe_mesh(
    nodes: list,
    elements: list,
    cell_types: list,
    file_path: str,
    point_data: Optional[dict] = None,
    element_data: Optional[dict] = None,
) -> None:
    with open(file_path, "w") as vtk_file:
        vtk_file.write("# vtk DataFile Version 4.2\n")
        vtk_file.write("FE Mesh Data\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET UNSTRUCTURED_GRID\n")

        # Write the node coordinates
        vtk_file.write(f"POINTS {len(nodes)} double\n")
        for node in nodes:
            vtk_file.write(f"{node[0]} {node[1]} {node[2]}\n")

        # Write the element connectivity
        num_elements = len(elements)
        total_num_entries = sum(
            len(cell) + 1 for cell in elements
        )  # Sum of nodes + 1 for the cell type
        vtk_file.write(f"\nCELLS {num_elements} {total_num_entries}\n")
        for element in elements:
            num_nodes = len(element)
            vtk_file.write(f"{num_nodes} {' '.join(str(node) for node in element)}\n")

        # Write the cell types
        vtk_file.write(f"\nCELL_TYPES {num_elements}\n")
        for cell_type in cell_types:
            vtk_file.write(f"{cell_type}\n")

        if point_data:
            vtk_file.write(f"\nPOINT_DATA {len(nodes)}\n")
            if "scalars" in point_data.keys():
                for pd in point_data["scalars"].keys():
                    vtk_file.write(f"\nSCALARS {pd} double 1\n")
                    vtk_file.write("LOOKUP_TABLE default\n")
                    for point in point_data["scalars"][pd]:
                        vtk_file.write(f"{point}\n")

            if "vectors" in point_data.keys():
                for vd in point_data["vectors"].keys():
                    vtk_file.write(f"\nVECTORS {vd} double\n")
                    for vector in point_data["vectors"][vd]:
                        for v in vector:
                            vtk_file.write(f"{v} ")
                        vtk_file.write("\n")

        if element_data:
            vtk_file.write(f"\nCELL_DATA {len(elements)}\n")
            if "scalars" in element_data.keys():
                for pd in element_data["scalars"].keys():
                    vtk_file.write(f"\nSCALARS {pd} double 1\n")
                    vtk_file.write("LOOKUP_TABLE default\n")
                    for point in element_data["scalars"][pd]:
                        vtk_file.write(f"{point}\n")

            if "vectors" in element_data.keys():
                for vd in element_data["vectors"].keys():
                    vtk_file.write(f"\nVECTORS {vd} double\n")
                    for vector in element_data["vectors"][vd]:
                        for v in vector:
                            vtk_file.write(f"{v} ")
                        vtk_file.write("\n")

    print(f"VTK FE mesh has been saved to {file_path}")
