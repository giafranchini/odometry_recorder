import numpy as np

def rotate(data : np.ndarray, axis : str, angle : float) -> np.ndarray:
    # angle > 0: counter-clockwise
    if axis == 'x':
        R = np.array([[1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

    return data @ R.T

def translate(data : np.ndarray, v : np.ndarray) -> np.ndarray:
    return data + v

def wrap_angle_2d(angle : float) -> float:
    return angle - 2.0 * np.pi * np.floor((angle + np.pi) / (2.0 * np.pi))

vec_wrap_angle_2d = np.vectorize(wrap_angle_2d)