import numpy as np
from os.path import expanduser
from math import pi, sin, cos

home = expanduser("~")

INPUT_PATH = f'{home}/ros2_iron_ws/src/odometry_recorder/data/emrs_localisation_roxy_loop_0/emrs_odometry_filtered_transf_sync.csv'
OUTPUT_PATH = f'{home}/ros2_iron_ws/src/odometry_recorder/data/emrs_localisation_roxy_loop_0/emrs_odometry_filtered_transf_sync_rotated.csv'

def read_data(path):
    file = open(path, 'r')
    return np.array([line.strip('\n\r').split(',') for i, line in enumerate(file.readlines()) if i != 0], dtype=float)

def write_data(path, data):
    np.savetxt(path, data, delimiter=",")

def rotate(data, axis, angle):
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

def translate(data, vector):
    return data + vector

def main():
    odom = read_data(INPUT_PATH)
    odom[:, [1, 2, 3]] = rotate(odom[:, [1, 2, 3]], 'z', -0.0873)
    odom[:, [1, 2, 3]] = translate(odom[:, [1, 2, 3]], np.array([0.485, 0.235, -0.77]))
    write_data(OUTPUT_PATH, odom)

if __name__ == '__main__':
    main()