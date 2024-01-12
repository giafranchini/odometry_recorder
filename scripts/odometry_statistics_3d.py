from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from os.path import expanduser
home = expanduser("~")

FUSE_ODOMETRY_PATH = f'{home}/iron_ws/src/odometry_recorder/data/fuse_odom.csv'
RL_ODOMETRY_PATH = f'{home}/iron_ws/src/odometry_recorder/data/rl_odom.csv'

def read_data(path):
    file = open(path, 'r')
    return np.array([line.strip('\n\r').split(',') for i, line in enumerate(file.readlines()) if i != 0], dtype=float)

def subsample(odom_data):
    # subsample data 1 per second
    timesteps = np.unique(odom_data[:,0])
    return np.array([odom_data[odom_data[:,0] == t].mean(axis=0) for t in timesteps])

def plot_residuals(first, second):
    res = first - second
    figure = plt.figure()

    ax0 = plt.subplot2grid((3,2), (0,0), rowspan=1, colspan=1)
    ax0.set_title('Residual X [m]')
    ax0.scatter(
        res[:,1], norm.pdf(res[:,1], res[:,1].mean(), res[:,1].std()), s=5, color='red', alpha=1
    )
    ax0.axvline(res[:,1].mean(), ymin=0, ymax=1, color='black', linestyle='dashed', label=f'Mean: {res[:,1].mean():.3f}')
    ax0.axvline(0, ymin=0, ymax=1, color='blue', linestyle='dashed', label='Zero mean')
    ax0.legend()

    ax1 = plt.subplot2grid((3,2), (1,0), rowspan=1, colspan=1)
    ax1.set_title('Residual Y [m]')
    ax1.scatter(
        res[:,2], norm.pdf(res[:,2], res[:,2].mean(), res[:,2].std()), s=5, color='red', alpha=1
    )
    ax1.axvline(res[:,2].mean(), ymin=0, ymax=1, color='black', linestyle='dashed', label=f'Mean: {res[:,2].mean():.3f}')
    ax1.axvline(0, ymin=0, ymax=1, color='blue', linestyle='dashed', label='Zero mean')
    ax1.legend()

    ax2 = plt.subplot2grid((3,2), (2,0), rowspan=1, colspan=1)
    ax2.set_title('Residual Z [m]')
    ax2.scatter(
        res[:,3], norm.pdf(res[:,3], res[:,3].mean(), res[:,3].std()), s=5, color='red', alpha=1
    )
    ax2.axvline(res[:,3].mean(), ymin=0, ymax=1, color='black', linestyle='dashed', label=f'Mean: {res[:,3].mean():.3f}')
    ax2.axvline(0, ymin=0, ymax=1, color='blue', linestyle='dashed', label='Zero mean')
    ax2.legend()

    ax3 = plt.subplot2grid((3,2), (0,1), rowspan=1, colspan=1)
    ax3.set_title('Residual Roll [rad]')
    ax3.scatter(
        res[:,4], norm.pdf(res[:,4], res[:,4].mean(), res[:,4].std()), s=5, color='red', alpha=1
    )
    ax3.axvline(res[:,4].mean(), ymin=0, ymax=1, color='black', linestyle='dashed', label=f'Mean: {res[:,4].mean():.3f}')
    ax3.axvline(0, ymin=0, ymax=1, color='blue', linestyle='dashed', label='Zero mean')
    ax3.legend()

    ax4 = plt.subplot2grid((3,2), (1,1), rowspan=1, colspan=1)
    ax4.set_title('Residual Pitch [rad]')
    ax4.scatter(
        res[:,5], norm.pdf(res[:,5], res[:,5].mean(), res[:,5].std()), s=5, color='red', alpha=1
    )
    ax4.axvline(res[:,5].mean(), ymin=0, ymax=1, color='black', linestyle='dashed', label=f'Mean: {res[:,5].mean():.3f}')
    ax4.axvline(0, ymin=0, ymax=1, color='blue', linestyle='dashed', label='Zero mean')
    ax4.legend()

    ax5 = plt.subplot2grid((3,2), (2,1), rowspan=1, colspan=1)
    ax5.set_title('Residual Yaw [rad]')
    ax5.scatter(
        res[:,6], norm.pdf(res[:,6], res[:,6].mean(), res[:,6].std()), s=5, color='red', alpha=1
    )
    ax5.axvline(res[:,6].mean(), ymin=0, ymax=1, color='black', linestyle='dashed', label=f'Mean: {res[:,6].mean():.3f}')
    ax5.axvline(0, ymin=0, ymax=1, color='blue', linestyle='dashed', label='Zero mean')
    ax5.legend()

    # d_euc = np.sqrt(np.square(res[:,1]) + np.square(res[:,2]))
    # ax3 = plt.subplot2grid((3,2), (0,1), rowspan=3, colspan=1)
    # ax3.set_title('Residual trajectory XY')
    # ax3.scatter(
    #     d_euc, norm.pdf(d_euc, d_euc.mean(), d_euc.std()), s=5, color='red', alpha=1
    # )
    # ax3.axvline(d_euc.mean(), ymin=0, ymax=1, color='black', linestyle='dashed', label=f'Mean: {res[:,3].mean():.3f}')
    # ax3.axvline(0, ymin=0, ymax=1, color='blue', linestyle='dashed', label='Zero mean')
    # ax3.legend()
    figure.tight_layout(h_pad=1.0)

    plt.show()

def plot_trajectory(first, second):

    ax = plt.figure().add_subplot(projection='3d')
    ax.set_title('Trajectory XYZ')
    ax.plot(first[:,1], first[:,2], first[:,3], color='red', alpha=1, label='RL odometry')
    ax.plot(second[:,1], second[:,2], second[:,3], color='blue', alpha=1, label='Fuse odometry')
    ax.legend()
    # ax.tight_layout(h_pad=1.0)

    plt.show()

def main():
    rl_odom   = read_data(RL_ODOMETRY_PATH)
    fuse_odom = read_data(FUSE_ODOMETRY_PATH)
    rl_odom   = subsample(np.delete(rl_odom, 0, axis=0))
    fuse_odom = subsample(np.delete(fuse_odom, 0, axis=0))

    t_diff = rl_odom.shape[0] - fuse_odom.shape[0]
    if t_diff > 0:
        rl_odom = rl_odom[:-t_diff]
    elif t_diff < 0:
        fuse_odom = fuse_odom[:-t_diff]

    plot_residuals(rl_odom, fuse_odom)
    plot_trajectory(rl_odom, fuse_odom)

if __name__ == '__main__':
    main()