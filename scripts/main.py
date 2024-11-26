
import numpy as np
from utils import read_data, load_params
from transform import rotate, translate
from plotting import *
from metrics import *

def main():
    params = load_params('/home/giacomo/ros2_iron_ws/src/odometry_recorder/scripts/params.yaml')
    trajectories = params['trajectories']
    transform = params['alignment']
    metrics = params['metrics']
    plotting = params['plotting']

    odometry_data = list()
    for _, value in trajectories.items():
        name = value[0]
        data_path = value[-1]
        odom = read_data(data_path)
        
        if transform['align'] and name != 'Ground Truth':
            yaw = transform['yaw']
            t = transform['translation']
            odom[:, [1, 2, 3]] = rotate(odom[:, [1, 2, 3]], 'z', yaw)
            odom[:, [1, 2, 3]] = translate(odom[:, [1, 2, 3]], t)
        
        odometry_data.append(odom)
    
    if plotting['plot']:
        # plot_motion_data(odometry_data, plot_velocities=True)
        # plot_trajectories(odometry_data, params=trajectories, two_d=False)
        plot_animated_trajectories(odometry_data, params=trajectories, two_d=False)
        # plot_residuals(odometry_data, plot_orientation=False)
        # plot_boxplot(odometry_data, plot_velocities=True)

    if (metrics['compute']):
        l = trajectory_length(odometry_data[-1])
        absolute_error = ate([odometry_data[0], odometry_data[-1]])
        relative_error = re_statistics(
            [odometry_data[0], odometry_data[-1]], 
            metrics['re_samples'], 
            int(np.ceil(metrics['re_interval'] * len(odometry_data[0])))
        )
        
        print(f"Calculating metrics between: {trajectories['traj0'][0]} odometry wrt {trajectories['traj1'][0]} odometry")
        print(f"\nTrajectory length: {l} [m]")
        print(f"\nAbsolute trajectory error: \n\tATE: {absolute_error} [m], ATE%: {(absolute_error / l) * 100} [%]")
        print(f"\nRelative trajectory error: \n\tMean: {relative_error[0]} [m], Std: {relative_error[1]} [m]")

if __name__ == '__main__':
    main()