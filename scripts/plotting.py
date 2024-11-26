import numpy as np

from scipy.stats import norm
from mpl_toolkits.mplot3d import art3d
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import check_shapes
from transform import vec_wrap_angle_2d

def plot_trajectories_2d(traj : list, params : dict) -> None:
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title('Trajectory on XY plane')
    
    drawed_start_point = False

    for t, key in zip(traj, params):
        t_name = params[key][0]
        t_color = params[key][1]
        t_path = params[key][-1]

        ax.plot(t[:,1], t[:,2], color=t_color, alpha=1, label=f'{t_name} odometry', linewidth=2)

        if not drawed_start_point:
            ax.text(t[0,1], t[0,2], 'Start', fontsize=12, color='black')
            drawed_start_point = True
        ax.text(t[-1,1], t[-1,2], 'End', fontsize=12, color='black')
  
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend()
    plt.show()
    

def plot_trajectories(traj : list, params : dict, two_d : bool = False) -> plt.Figure:
    if two_d:
        plot_trajectories_2d(traj, params)
        return
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    drawed_start_point = False

    for t, key in zip(traj, params):
        t_name = params[key][0]
        t_color = params[key][1]
        t_path = params[key][-1]

        ax.plot(t[:,1], t[:,2], t[:,3], color=t_color, alpha=1, label=f'{t_name} odometry', linewidth=2)

        if not drawed_start_point:
            ax.text(t[0,1], t[0,2], t[0,3] - 0.1, 'Start', fontsize=12, color='black')
            drawed_start_point = True
        ax.text(t[-1,1], t[-1,2], t[-1,3], 'End', fontsize=12, color='black')
    
    ax.set_xlim(traj[-1][:, 1].min() - 0.5, traj[-1][:, 1].max() + 0.5)
    ax.set_ylim(traj[-1][:, 2].min() - 0.5, traj[-1][:, 2].max() + 0.5)
    ax.set_zlim(traj[-1][:, 3].min() - 0.25, traj[-1][:, 3].max() + 1.0)
   
    ax.set_title('3D estimated rajectory vs ground truth')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()

    # Beautify the plot
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    return fig

def plot_motion_data(data : list, plot_velocities : bool = False, crop_index : int = -1) -> None:
    """
    Plots: 
        X vs Time, Y vs Time, Z vs Time, Roll vs Time, Pitch vs Time, Yaw vs Time and
        Vx vs Time, Vy vs Time, Vz vs Time, Vroll vs Time, Vpitch vs Time, Vyaw vs Time, for two datasets.
    
    Parameters:
    - data[0]: numpy.ndarray, first dataset with columns [time, x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz].
    - data[1]: numpy.ndarray, second dataset with columns [time, x, y, z, roll, pitch, yaw, , vx, vy, vz, wx, wy, wz].
    """

    # Check if data has the correct shape
    data = check_shapes(data)

    if plot_velocities:
        if data[0].shape[1] < 13 or data[1].shape[1] < 13:
            raise ValueError("Each input array must have at least 13 columns: [time, x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz].")
    else:
        if data[0].shape[1] < 7 or data[1].shape[1] < 7:
            raise ValueError("Each input array must have at least 7 columns: [time, x, y, z, roll, pitch, yaw].")
        
    # Extracting trajectory data, crop end of data to have only crop_index samples (defaults to -1, which means no cropping)
    time1, x1, y1, z1, roll1, pitch1, yaw1 = data[0][:, 0][:crop_index], data[0][:, 1][:crop_index], data[0][:, 2][:crop_index], data[0][:, 3][:crop_index], -data[0][:, 4][:crop_index], -data[0][:, 5][:crop_index], vec_wrap_angle_2d(data[0][:, 6][:crop_index])
    time2, x2, y2, z2, roll2, pitch2, yaw2 = data[1][:, 0][:crop_index], data[1][:, 1][:crop_index], data[1][:, 2][:crop_index], data[1][:, 3][:crop_index], -data[1][:, 4][:crop_index], -data[0][:, 5][:crop_index], vec_wrap_angle_2d(data[0][:, 6][:crop_index])
    
    yaw1 = np.pi - yaw1
    yaw2 = np.pi - yaw2
    
    # Creating subplots
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    # fig.suptitle("Chassis pose over time")
    t1 = np.array(range(time1.shape[0])) / 100
    t2 = np.array(range(time2.shape[0])) / 100

    # Plot X vs Time
    axs[0, 0].plot(t1, x1, label="Estimated", color="red")
    axs[0, 0].plot(t2, x2, label="Ground truth", color="blue")
    axs[0, 0].set_title("X vs Time")
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel("X [m]")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot Y vs Time
    axs[1, 0].plot(t1, y1, label="Estimated", color="red")
    axs[1, 0].plot(t2, y2, label="Ground truth", color="blue")
    axs[1, 0].set_title("Y vs Time")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Y [m]")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plot Z vs Time
    axs[2, 0].plot(t1, z1, label="Estimated", color="red")
    axs[2, 0].plot(t2, z2, label="Ground truth", color="blue")
    axs[2, 0].set_title("Z vs Time")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].set_ylabel("Z [m]")
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    
    # Plot Roll vs Time
    axs[0, 1].plot(t1, roll1, label="Estimated", color="red")
    axs[0, 1].plot(t2, roll2, label="Ground truth", color="blue")
    axs[0, 1].set_title("Roll vs Time")
    axs[0, 1].set_xlabel("Time [s]")
    axs[0, 1].set_ylabel("Roll [rad]")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot Pitch vs Time
    axs[1, 1].plot(t1, pitch1, label="Estimated", color="red")
    axs[1, 1].plot(t2, pitch2, label="Ground truth", color="blue")
    axs[1, 1].set_title("Pitch vs Time")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Pitch [rad]")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Plot Yaw vs Time
    axs[2, 1].plot(t1, yaw1, label="Estimated", color="red")
    axs[2, 1].plot(t2, yaw2, label="Ground truth", color="blue")
    axs[2, 1].set_title("Yaw vs Time")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].set_ylabel("Yaw [rad]")
    axs[2, 1].legend()
    axs[2, 1].grid(True)
        
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

    if plot_velocities:
        # Extracting velocity data, crop end of data to have only crop_index samples (defaults to -1, which means no cropping)
        vx1, vy1, vz1, vroll1, vpitch1, vyaw1 = data[0][:, 7][:crop_index], data[0][:, 8][:crop_index], data[0][:, 9][:crop_index], data[0][:, 10][:crop_index], data[0][:, 11][:crop_index], data[0][:, 12][:crop_index]
        vx2, vy2, vz2, vroll2, vpitch2, vyaw2 = data[1][:, 7][:crop_index], data[1][:, 8][:crop_index], data[1][:, 9][:crop_index], data[1][:, 10][:crop_index], data[1][:, 11][:crop_index], data[1][:, 12][:crop_index]

        # invert points (this is some for chrono, probably Z is pointing downwars)
        vx2 = -vx2
        vy2 = -vy2
        vroll2 = -vroll2
        vpitch2 = -vpitch2

        # Creating subplots
        fig, axs = plt.subplots(3, 2, figsize=(12, 8))
        # fig.suptitle("Chassis velocities over time")

        # Plot Vx vs Time
        axs[0, 0].plot(t1, vx1, label="Estimated", color="red")
        axs[0, 0].plot(t2, vx2, label="Ground truth", color="blue")
        axs[0, 0].set_title("Vx vs Time")
        axs[0, 0].set_xlabel("Time [s]")
        axs[0, 0].set_ylabel("Vx [m/s]")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot Vy vs Time
        axs[1, 0].plot(t1, vy1, label="Estimated", color="red")
        axs[1, 0].plot(t2, vy2, label="Ground truth", color="blue")
        axs[1, 0].set_title("Vy vs Time")
        axs[1, 0].set_xlabel("Time [s]")
        axs[1, 0].set_ylabel("Vy [m/s]")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # Plot Vz vs Time
        axs[2, 0].plot(t1, vz1, label="Estimated", color="red")
        axs[2, 0].plot(t2, vz2, label="Ground truth", color="blue")
        axs[2, 0].set_title("Vz vs Time")
        axs[2, 0].set_xlabel("Time [s]")
        axs[2, 0].set_ylabel("Vz [m/s]")
        axs[2, 0].legend()
        axs[2, 0].grid(True)

        # Plot Vroll vs Time
        axs[0, 1].plot(t1, vroll1, label="Estimated", color="red")
        axs[0, 1].plot(t2, vroll2, label="Ground truth", color="blue")
        axs[0, 1].set_title("Vroll vs Time")
        axs[0, 1].set_xlabel("Time [s]")
        axs[0, 1].set_ylabel("Vroll [rad/s]")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # Plot Vpitch vs Time
        axs[1, 1].plot(t1, vpitch1, label="Estimated", color="red")
        axs[1, 1].plot(t2, vpitch2, label="Ground truth", color="blue")
        axs[1, 1].set_title("Vpitch vs Time")
        axs[1, 1].set_xlabel("Time [s]")
        axs[1, 1].set_ylabel("Vpitch [rad/s]")
        axs[1, 1].legend()
        axs[1, 1].grid(True)

        # Plot Vyaw vs Time
        axs[2, 1].plot(t1, vyaw1, label="Estimated", color="red")
        axs[2, 1].plot(t2, vyaw2, label="Ground truth", color="blue")
        axs[2, 1].set_title("Vyaw vs Time")
        axs[2, 1].set_xlabel("Time [s]")
        axs[2, 1].set_ylabel("Vyaw [rad/s]")
        axs[2, 1].legend()
        axs[2, 1].grid(True)

        # Adjust layout and show plot
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        
def plot_animated_trajectories(traj : list, params : dict, two_d : bool = False) -> None:
    if two_d:
        # plot_animated_trajectories_2d(traj)
        return
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    drawed_start_point = False

    for t, key in zip(traj, params):
        t_name = params[key][0]
        t_color = params[key][1]
        t_path = params[key][-1]

        ax.plot(t[:,1], t[:,2], t[:,3], color=t_color, alpha=1, label=f'{t_name} Trajectory', linewidth=2)
        if not drawed_start_point:
            ax.text(t[0,1], t[0,2], t[0,3] - 0.1, 'Start', fontsize=12, color='black')
            drawed_start_point = True
        ax.text(t[-1,1], t[-1,2], t[-1,3], 'End', fontsize=12, color='black')
    
    # Update function for animation
    def rotate_azimut(frame : int) -> art3d.Line3D:
        ax.view_init(elev=10., azim=-180+frame)
        return fig,
    
    ax.set_xlim(traj[-1][:, 1].min() - 0.5, traj[-1][:, 1].max() + 0.5)
    ax.set_ylim(traj[-1][:, 2].min() - 0.5, traj[-1][:, 2].max() + 0.5)
    ax.set_zlim(traj[-1][:, 3].min() - 0.25, traj[-1][:, 3].max() + 1.0)
   
    # ax.set_title('Estimated vs GNSS trajectory')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()

    # Beautify the plot
    ax.grid(True)
    plt.tight_layout()

    ani = FuncAnimation(fig, rotate_azimut, frames=180)
    output_animation_path = "/home/giacomo/trajectory_animation.mp4"
    ani.save(output_animation_path, fps=10)

def plot_residuals(data : list, plot_orientation : bool = False) -> None:
    data = check_shapes(data)
    first = data[0]
    second = data[1]
    figure = plt.figure()

    residual_pos = first[:, 1:4] - second[:, 1:4]
    ax0 = plt.subplot2grid((3,2), (0,0), rowspan=1, colspan=1)
    ax0.set_title('Residual X [m]')
    ax0.scatter(
        residual_pos[:,0], norm.pdf(residual_pos[:,0], residual_pos[:,0].mean(), residual_pos[:,0].std()), s=5, color='red', alpha=1
    )
    ax0.axvline(residual_pos[:,0].mean(), ymin=0, ymax=1, color='black', linestyle='dashed', label=f'Mean: {residual_pos[:,0].mean():.3f}')
    ax0.axvline(0, ymin=0, ymax=1, color='blue', linestyle='dashed', label='Zero mean')
    ax0.legend()

    ax1 = plt.subplot2grid((3,2), (1,0), rowspan=1, colspan=1)
    ax1.set_title('Residual Y [m]')
    ax1.scatter(
        residual_pos[:,1], norm.pdf(residual_pos[:,1], residual_pos[:,1].mean(), residual_pos[:,1].std()), s=5, color='red', alpha=1
    )
    ax1.axvline(residual_pos[:,1].mean(), ymin=0, ymax=1, color='black', linestyle='dashed', label=f'Mean: {residual_pos[:,1].mean():.3f}')
    ax1.axvline(0, ymin=0, ymax=1, color='blue', linestyle='dashed', label='Zero mean')
    ax1.legend()

    ax2 = plt.subplot2grid((3,2), (2,0), rowspan=1, colspan=1)
    ax2.set_title('Residual Z [m]')
    ax2.scatter(
        residual_pos[:,2], norm.pdf(residual_pos[:,2], residual_pos[:,2].mean(), residual_pos[:,2].std()), s=5, color='red', alpha=1
    )
    ax2.axvline(residual_pos[:,2].mean(), ymin=0, ymax=1, color='black', linestyle='dashed', label=f'Mean: {residual_pos[:,2].mean():.3f}')
    ax2.axvline(0, ymin=0, ymax=1, color='blue', linestyle='dashed', label='Zero mean')
    ax2.legend()

    if plot_orientation:
        if first.shape[1] != 6 and second.shape[1] != 6:
            raise Exception('Orientation data not in RPY format, please check the data')
        residual_rot = first[:, 4:] - second[:, 4:]
        ax3 = plt.subplot2grid((3,2), (0,1), rowspan=1, colspan=1)
        ax3.set_title('Residual Roll [rad]')
        ax3.scatter(
            residual_rot[:,0], norm.pdf(residual_rot[:,0], residual_rot[:,0].mean(), residual_rot[:,0].std()), s=5, color='red', alpha=1
        )
        ax3.axvline(residual_rot[:,0].mean(), ymin=0, ymax=1, color='black', linestyle='dashed', label=f'Mean: {residual_rot[:,0].mean():.3f}')
        ax3.axvline(0, ymin=0, ymax=1, color='blue', linestyle='dashed', label='Zero mean')
        ax3.legend()

        ax4 = plt.subplot2grid((3,2), (1,1), rowspan=1, colspan=1)
        ax4.set_title('Residual Pitch [rad]')
        ax4.scatter(
            residual_rot[:,1], norm.pdf(residual_rot[:,1], residual_rot[:,1].mean(), residual_rot[:,1].std()), s=5, color='red', alpha=1
        )
        ax4.axvline(residual_rot[:,1].mean(), ymin=0, ymax=1, color='black', linestyle='dashed', label=f'Mean: {residual_rot[:,1].mean():.3f}')
        ax4.axvline(0, ymin=0, ymax=1, color='blue', linestyle='dashed', label='Zero mean')
        ax4.legend()

        ax5 = plt.subplot2grid((3,2), (2,1), rowspan=1, colspan=1)
        ax5.set_title('Residual Yaw [rad]')
        ax5.scatter(
            residual_rot[:,2], norm.pdf(residual_rot[:,2], residual_rot[:,2].mean(), residual_rot[:,2].std()), s=5, color='red', alpha=1
        )
        ax5.axvline(residual_rot[:,2].mean(), ymin=0, ymax=1, color='black', linestyle='dashed', label=f'Mean: {residual_rot[:,2].mean():.3f}')
        ax5.axvline(0, ymin=0, ymax=1, color='blue', linestyle='dashed', label='Zero mean')
        ax5.legend()

    figure.tight_layout(h_pad=1.0)
    plt.show()

def plot_boxplot(data : list, plot_velocities : bool = False) -> None:
    data = check_shapes(data)

    if plot_velocities:
        if data[0].shape[1] < 13 or data[1].shape[1] < 13:
            raise ValueError("Each input array must have at least 13 columns: [time, x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz].")
    else:
        if data[0].shape[1] < 7 or data[1].shape[1] < 7:
            raise ValueError("Each input array must have at least 7 columns: [time, x, y, z, roll, pitch, yaw].")
        
    first = data[0]
    second = data[1]
    
    labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw']
    colors = 6 * ['red']
    ylabel_pos = '[m]'
    ylabel_rot = '[rad]'
    ylabel_lin_vel = '[m/s]'
    ylabel_ang_vel = '[rad/s]'
    ylabel = 'Observed values'

    # Compute residuals
    res_x = first[:, 1] - second[:, 1]
    res_y = first[:, 2] - second[:, 2]
    res_z = first[:, 3] - second[:, 3]
    res_roll = first[:, 4] - second[:, 4]
    res_pitch = first[:, 5] - second[:, 5]
    res_yaw = vec_wrap_angle_2d(first[:, 6]) - vec_wrap_angle_2d(second[:, 6])
    res = np.column_stack((res_x, res_y, res_z, res_roll, res_pitch, res_yaw))

    # Create 3x2 grid of subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 8))
    fig.suptitle('Chassis Pose Residuals')

    for i, ax in enumerate(axs.T.ravel()):
        # Plot boxplot for each pose residual
        bplot = ax.boxplot(res[:, i], patch_artist=True, labels=[labels[i]], meanline=True, showmeans=True)
        bplot['medians'][0].set(color='black', linewidth=2)
        bplot['means'][0].set(color='black', linestyle='--', linewidth=2)
        
        # Set box color
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[i])
        
        # Adding grid lines
        ax.yaxis.grid(True)
        ax.set_ylabel(ylabel + ' ' + ((ylabel_pos if i < 3 else ylabel_rot)))
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    if plot_velocities:
        labels = ['Vx', 'Vy', 'Vz', 'Vroll', 'Vpitch', 'Vyaw']
        colors = 6 * ['blue']

        # Compute residuals
        res = first[:, 7:] - second[:, 7:]

        # Create 3x2 grid of subplots
        fig, axs = plt.subplots(3, 2, figsize=(10, 8))
        fig.suptitle('Chassis velocities residuals boxplot')

        for i, ax in enumerate(axs.flat):
            # Plot boxplot for each velocity residual
            bplot = ax.boxplot(res[:, i], patch_artist=True, labels=[labels[i]], meanline=True, showmeans=True)
            bplot['medians'][0].set(color='black', linewidth=2)
            bplot['means'][0].set(color='black', linestyle='--', linewidth=2)

            # Set box color
            for patch in bplot['boxes']:
                patch.set_facecolor(colors[i])

            # Adding grid lines
            ax.yaxis.grid(True)
            ax.set_ylabel(ylabel + ' ' + ((ylabel_pos if i < 3 else ylabel_rot)))

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()