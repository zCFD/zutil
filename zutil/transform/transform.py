import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr
import numpy as np


def scale(scale_factors: list[float], A2B: np.ndarray = None) -> np.ndarray:
    if A2B is None:
        A2B = np.identity(4)

    scale_matrix = np.identity(4)

    for ii in range(3):
        scale_matrix[ii][ii] = scale_factors[ii]

    B2C = np.matmul(scale_matrix, A2B)

    return B2C


def translate(
    point_b: list[float], point_c: list[float], A2B: np.ndarray = None
) -> np.ndarray:
    if A2B is None:
        A2B = np.identity(4)

    translation_matrix = np.identity(4)
    translation_vector = np.asarray(point_c) - np.asarray(point_b)

    for ii in range(3):
        translation_matrix[ii][3] = translation_vector[ii]

    B2C = np.matmul(translation_matrix, A2B)

    return B2C


def rotate(axis: list[float], angle_deg: float, A2B: np.ndarray = None) -> np.ndarray:
    if A2B is None:
        A2B = np.identity(4)

    angle_rad = np.deg2rad(angle_deg)
    a = np.asarray(list(axis) + [angle_rad])
    rotation_matrix = pyrot.matrix_from_axis_angle(a)

    B2C = np.identity(4)
    B2C = pytr.rotate_transform(B2C, rotation_matrix)

    C2D = np.matmul(B2C, A2B)

    return C2D
