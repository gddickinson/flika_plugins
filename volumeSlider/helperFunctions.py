import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
from numpy import moveaxis
from skimage.transform import rescale
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
from flika import global_vars as g
from pathlib import Path
from typing import List, Tuple
import time
from functools import wraps
import logging

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Use perf_counter for more precise timing
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logging.info(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

##############       Helper functions        ###########################################

#============GUI SETUP===========================================================
def windowGeometry(instance, left: int = 300, top: int = 300, height: int = 600, width: int = 400) -> None:
    """
    Set the geometry of a window.

    Args:
        instance: The window instance.
        left: Left coordinate of the window.
        top: Top coordinate of the window.
        height: Height of the window.
        width: Width of the window.
    """
    instance.left = left
    instance.top = top
    instance.height = height
    instance.width = width

def setSliderUp(slider: QtWidgets.QSlider, minimum: int = 0, maximum: int = 100,
                tickInterval: int = 1, singleStep: int = 1, value: int = 0) -> None:
    """
    Set up a QSlider with given parameters.

    Args:
        slider: The QSlider instance.
        minimum: Minimum value of the slider.
        maximum: Maximum value of the slider.
        tickInterval: Interval between ticks.
        singleStep: Step size for single increments.
        value: Initial value of the slider.
    """
    slider.setFocusPolicy(QtCore.Qt.StrongFocus)
    slider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
    slider.setRange(minimum, maximum)
    slider.setTickInterval(tickInterval)
    slider.setSingleStep(singleStep)
    slider.setValue(value)

def setMenuUp(menuItem: QtWidgets.QAction, menu: QtWidgets.QMenu, shortcut: str = None,
              statusTip: str = '', connection: callable = None) -> None:
    """
    Set up a menu item with given parameters.

    Args:
        menuItem: The QAction instance.
        menu: The QMenu to add the action to.
        shortcut: Keyboard shortcut for the action.
        statusTip: Status tip to display.
        connection: Function to connect to the triggered signal.
    """
    if shortcut:
        menuItem.setShortcut(shortcut)
    menuItem.setStatusTip(statusTip)
    if connection:
        menuItem.triggered.connect(connection)
    menu.addAction(menuItem)
#=================================================================================


def get_transformation_matrix(theta: float) -> np.ndarray:
    """
    Calculate the transformation matrix for a given angle.

    Args:
        theta: Angle in degrees.

    Returns:
        3x3 transformation matrix.
    """
    theta_rad = np.radians(theta)  # Convert to radians
    hx = np.cos(theta_rad)
    sy = np.sin(theta_rad)

    return np.array([
        [1, hx, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])


@profile
def get_transformation_coordinates(I: np.ndarray, theta: float) -> tuple:
    """
    Get transformation coordinates for an image given an angle.

    Args:
        I: Input image array.
        theta: Angle in degrees.

    Returns:
        Tuple of old and new coordinates.
    """
    S = get_transformation_matrix(theta)
    S_inv = np.linalg.inv(S)
    mx, my = I.shape

    corners = np.array([[0, 0, mx, mx],
                        [0, my, 0, my],
                        [1, 1, 1, 1]])

    transformed_corners = np.matmul(S, corners)[:-1, :]

    range_x = np.round(np.array([np.min(transformed_corners[0]), np.max(transformed_corners[0])])).astype(int)
    range_y = np.round(np.array([np.min(transformed_corners[1]), np.max(transformed_corners[1])])).astype(int)

    new_coords = np.meshgrid(np.arange(range_x[0], range_x[1]), np.arange(range_y[0], range_y[1]))
    new_coords = [coord.flatten() for coord in new_coords]

    new_homog_coords = np.vstack([new_coords[0], new_coords[1], np.ones(len(new_coords[0]))])
    old_coords = np.matmul(S_inv, new_homog_coords)[:-1, :]

    old_coords[old_coords >= np.array([mx-1, my-1])[:, np.newaxis]] = -1
    old_coords[old_coords < 1] = -1

    keep_coords = np.all(old_coords != -1, axis=0)
    new_coords = [coord[keep_coords] for coord in new_coords]
    old_coords = [coord[keep_coords] for coord in old_coords]

    return old_coords, new_coords


def perform_shear_transform(A: np.ndarray, shift_factor: float, interpolate: bool,
                            datatype: np.dtype, theta: float,
                            inputArrayOrder: List[int] = [0, 3, 1, 2],
                            displayArrayOrder: List[int] = [3, 0, 1, 2]) -> np.ndarray:
    """
    Perform a shear transform on the input array.

    Args:
        A: Input 4D array.
        shift_factor: Factor to shift the array.
        interpolate: Whether to use interpolation.
        datatype: Data type of the output array.
        theta: Angle for the shear transform.
        inputArrayOrder: Order of dimensions for input array.
        displayArrayOrder: Order of dimensions for output array.

    Returns:
        Transformed 4D array.
    """
    A = np.moveaxis(A, inputArrayOrder, [0, 1, 2, 3])
    m1, m2, m3, m4 = A.shape

    if interpolate:
        A_rescaled = np.zeros((m1*int(shift_factor), m2, m3, m4), dtype=datatype)
        for v in range(m4):
            logging.info(f'Upsampling Volume #{v+1}/{m4}')
            g.m.statusBar().showMessage(f'Upsampling Volume #{v+1}/{m4}')
            A_rescaled[:, :, :, v] = rescale(A[:, :, :, v], (shift_factor, 1.), mode='constant', preserve_range=True)
    else:
        A_rescaled = np.repeat(A, shift_factor, axis=1)

    mx, my, mz, mt = A_rescaled.shape
    I = A_rescaled[:, :, 0, 0]
    old_coords, new_coords = get_transformation_coordinates(I, theta)
    old_coords = np.round(old_coords).astype(int)
    new_mx, new_my = np.max(new_coords[0]) + 1, np.max(new_coords[1]) + 1

    D = np.zeros((new_mx, new_my, mz, mt), dtype=datatype)
    D[new_coords[0], new_coords[1], :, :] = A_rescaled[old_coords[0], old_coords[1], :, :]
    E = np.moveaxis(D, [0, 1, 2, 3], displayArrayOrder)

    return E

def getCorners(vol: np.ndarray) -> np.ndarray:
    """
    Get the corners of a volume.

    Args:
        vol: Input 3D array.

    Returns:
        Array with corners marked.
    """
    z, x, y = vol.nonzero()
    z_min, z_max = np.min(z), np.max(z)
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    newArray = np.zeros(vol.shape, dtype=bool)

    corner_coords = [
        (z_min, x_min, y_min), (z_min, x_max, y_min),
        (z_min, x_min, y_max), (z_min, x_max, y_max),
        (z_max, x_min, y_min), (z_max, x_max, y_min),
        (z_max, x_min, y_max), (z_max, x_max, y_max)
    ]

    for coord in corner_coords:
        newArray[coord] = True

    return newArray

def getDimensions(vol: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """
    Get the dimensions of a volume.

    Args:
        vol: Input 3D array.

    Returns:
        Tuple of (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    z, x, y = vol.nonzero()
    return np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)

def getMaxDimension(vol: np.ndarray) -> int:
    """
    Get the maximum dimension of a volume.

    Args:
        vol: Input 3D array.

    Returns:
        Maximum dimension value.
    """
    z, x, y = vol.nonzero()
    return np.max([np.max(x), np.max(y), np.max(z)])


def plot_cube(ax: plt.Axes, cube_definition: list) -> None:
    """
    Plot a cube on a 3D axis.

    Args:
        ax: Matplotlib 3D axis.
        cube_definition: List of 3D coordinates defining the cube.
    """
    cube_definition_array = np.array([list(item) for item in cube_definition])

    points = cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0]
    ]

    points = np.vstack((
        points,
        points[0] + vectors[0] + vectors[1],
        points[0] + vectors[0] + vectors[2],
        points[0] + vectors[1] + vectors[2],
        points[0] + vectors[0] + vectors[1] + vectors[2]
    ))

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]]
    ]

    faces = Poly3DCollection(edges, linewidths=1, edgecolors='k')
    faces.set_facecolor((0, 0, 0.1, 0.1))

    ax.add_collection3d(faces)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)
    ax.set_aspect('equal')



def shorten_path(file_path: str, length: int) -> str:
    """
    Shorten a file path to include only the last 'length' parts.

    Args:
        file_path: The full file path.
        length: The number of path parts to include.

    Returns:
        Shortened file path.
    """
    return str(Path(*Path(file_path).parts[-length:]))
