import matplotlib

matplotlib.use("Agg")  # 必须在 import pyplot 之前设置！
import matplotlib.pyplot as plt
from evo.tools import plot


def vis_trajs(traj_dict, xyz_ranges=None, save_path=None, vis=True):
    fig = plt.figure()
    plot.trajectories(fig, traj_dict, plot.PlotMode.xyz)

    if xyz_ranges is not None:
        ax = fig.axes[0]
        # 设置坐标轴范围
        ax.set_xlim([xyz_ranges[0], xyz_ranges[1]])
        ax.set_ylim([xyz_ranges[0], xyz_ranges[1]])
        ax.set_zlim([xyz_ranges[0], xyz_ranges[1]])
    plt.savefig(save_path)
    plt.close()
