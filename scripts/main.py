
import numpy as np
from utils import read_data, TRAJECTORY_PATHS, TRAJECTORY_NAMES
from transform import rotate, translate
from plotting import plot_trajectories, plot_residuals, plot_boxplot
from metrics import ate, re_statistics, trajectory_length

SOURCE = 1 # 0 for robot_localization, 1 for fuse

def main():
    # First we need to transform the odometry data from the odom frame to the local cartesian frame 
    t = np.array([0.485, 0.235, -0.77])
    yaw = -0.0873 
    odometry_data = list()
    for i in range(len(TRAJECTORY_NAMES)):
        odom = read_data(TRAJECTORY_PATHS[i])
        
        if TRAJECTORY_NAMES[i] != 'gnss':
            odom[:, [1, 2, 3]] = rotate(odom[:, [1, 2, 3]], 'z', yaw)
            odom[:, [1, 2, 3]] = translate(odom[:, [1, 2, 3]], t)
        odometry_data.append(odom)
    # plot_trajectories(odometry_data, two_d=False)
    # plot_trajectories(odometry_data, two_d=True)
    # plot_residuals(odometry_data[1:], plot_orientation=False)
    # plot_boxplot(odometry_data[1:], plot_orientation=False)
    motion_model = 'omnidirectional_3d' if 'omnidirectional_3d' in TRAJECTORY_PATHS[0] else 'emrs' 
    print("Robot localization motion model: omnidirectional_3d")
    print(f"Fuse motion model: {motion_model}")
    print(f"Calculating metrics for odometry source {TRAJECTORY_NAMES[SOURCE]} wrt {TRAJECTORY_NAMES[2]}")
    l = trajectory_length(odometry_data[2])
    traj_error = ate([odometry_data[SOURCE], odometry_data[2]])
    relative_traj_error = re_statistics([odometry_data[SOURCE], odometry_data[2]], 10000, int(0.05 * l))
    print(f"Trajectory length: {l} [m]")
    print(f"Absolute trajectory error: \n\tATE: {traj_error} [m], ATE%: {(traj_error / l) * 100} [%]")
    print(f"Relative trajectory error: \n\tMean: {relative_traj_error[0]} [m], Std: {relative_traj_error[1]} [m]")

if __name__ == '__main__':
    main()