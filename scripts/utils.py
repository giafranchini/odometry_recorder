import numpy as np
from os.path import expanduser

TRAJECTORY_NAMES = ['robot_localization', 'fuse', 'gnss']
COLORS = ['red', 'blue', 'green']
TRAJECTORY_PATHS = [expanduser("~") + '/ros2_iron_ws/src/odometry_recorder/data/' + path for path in 
    [
        'emrs_localisation_roxy_loop_1/emrs/emrs_odometry_filtered_rl_transf_sync.csv', 
        'emrs_localisation_roxy_loop_1/emrs/emrs_odometry_filtered_transf_sync.csv', 
        'emrs_localisation_roxy_loop_1/emrs/gnss_odom_sync.csv'
    ]
]

def read_data(path : str) -> np.ndarray:
  file = open(path, 'r')
  return np.array([line.strip('\n\r').split(',') for i, line in enumerate(file.readlines()) if i != 0], dtype=float)

def write_data(path : str, data : list) -> None:
  np.savetxt(path, data, delimiter=",")

def check_shapes(data : list) -> list:
  if len(data) != 2:
    raise Exception('Only two odometry data arrays are allowed')
  first = np.array(data[0])
  second = np.array(data[1])
  if first.shape[0] != second.shape[0]:
    if first.shape[0] > second.shape[0]:
      second = np.append(second, [second[-1]], axis=0)
    else:
      first = np.append(first, [first[-1]], axis=0)
    data[0] = first
    data[1] = second
  return data
