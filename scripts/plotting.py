from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from utils import check_shapes, COLORS, TRAJECTORY_NAMES

def plot_trajectories(traj : list, two_d : bool = False) -> None:
    if two_d:
        plot_trajectories_2d(traj)
        return
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_title('Trajectory XYZ')
    for i, t in enumerate(traj):
        ax.plot(t[:,1], t[:,2], t[:,3], color=COLORS[i], alpha=1, label=f'{TRAJECTORY_NAMES[i]} odometry')
        ax.text(t[0,1], t[0,2], t[0,3], 'Start', fontsize=12, color='black')
        ax.text(t[-1,1], t[-1,2], t[-1,3], 'End', fontsize=12, color='black')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()
    plt.show()

def plot_trajectories_2d(traj : list) -> None:
    ax = plt.figure().add_subplot()
    ax.set_title('Trajectory XY')
    for i, t in enumerate(traj):
        ax.plot(t[:,1], t[:,2], color=COLORS[i], alpha=1, label=f'{TRAJECTORY_NAMES[i]} odometry')
        ax.text(t[0,1], t[0,2], 'Start', fontsize=12, color='black')
        ax.text(t[-1,1], t[-1,2], 'End', fontsize=12, color='black')    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend()
    plt.show()

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

def plot_boxplot(data : list, plot_orientation : bool = False) -> None:
    data = check_shapes(data)
    first = data[0]
    second = data[1]
    _, (ax) = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    
    labels = ['X', 'Y', 'Z']
    colors = ['pink', 'lightblue', 'lightgreen']
    xlabel = 'Residuals [m]'
    ylabel = 'Observed values'
    res = first[:, 1:4] - second[:, 1:4]
    
    if plot_orientation:
        labels += ['Roll', 'Pitch', 'Yaw']
        colors += ['lightyellow', 'lightgrey', 'lightcoral']
        xlabel = 'Residuals [m, rad]'
        res = np.append(res, first[:, 4:] - second[:, 4:], axis=1)    

    bplot = ax.boxplot(res, patch_artist=True, labels=labels)
    ax.set_title('Residuals boxplot')

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    # adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel) 

    plt.show()