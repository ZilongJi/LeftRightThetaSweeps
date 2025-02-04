import numpy as np
import matplotlib.pyplot as plt
import brainpy.math as bm
from matplotlib.animation import FuncAnimation

def Animate_GC_bump(Grid_net, bumpactivity, animal_pos, savefolder):
    n_step = 10
    phi = Grid_net.value_grid
    transform = bm.array([[1, 0], [0, 1]])
    pos_mec = bm.matmul(transform, phi.T).T

    x_min, x_max = np.min(pos_mec[:, 0]), np.max(pos_mec[:, 0])
    y_min, y_max = np.min(pos_mec[:, 1]), np.max(pos_mec[:, 1])

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    fig, ax_ani = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    ax_ani.set_aspect("equal")
    ax_ani.grid(True)
    # ax_ani.scatter(
    #     center_grid[start_time:, 0],
    #     center_grid[start_time:, 1],
    #     c=max_r_grid[start_time:],
    #     cmap="viridis",
    #     s=10,
    # )
    # fig.colorbar(sc, ax=ax_ani)
    data = bumpactivity[::n_step, :]
    T = data.shape[0]

    ax_ani.plot(animal_pos[0], animal_pos[1], color="black")

    vmin1 = 0
    vmax1 = np.max(data)
    scatter1 = ax_ani.scatter([], [], c=[], s=200, cmap="Blues", vmin=vmin1, vmax=vmax1)
    ax_ani.set_title("Bump activity")
    ax_ani.set_xlim(x_min, x_max)
    ax_ani.set_ylim(y_min, y_max)

    def update(frame):
        y1 = data[frame].flatten()
        scatter1.set_offsets(np.column_stack((pos_mec[:, 0], pos_mec[:, 1])))
        scatter1.set_array(y1)
        return scatter1

    ani = FuncAnimation(fig, update, frames=T, interval=20, blit=False)

    plt.tight_layout()

    aniname = savefolder + "grid_activities.gif"
    ani.save(aniname, writer="Pillow", fps=30)
    
def Animate_HD_bump(HD_net, r, savefolder):
    n_step = 2
    x1 = HD_net.x
    data1 = r[::n_step, :]
    # data4 = r_sen[:, center_sensory_index]
    N = data1.shape[1]
    T = data1.shape[0]
    # 创建画布和轴
    fig, ax_ani = plt.subplots()
    ax_ani.set_ylim(-0.1, 1.5)
    ax_ani.set_xlim(-np.pi, np.pi)
    # 创建初始空白线条
    (line1,) = ax_ani.plot([], [])

    # 更新线条的函数
    def update(frame):
        y1 = data1[frame].flatten()
        line1.set_data(x1, y1)
        return line1

    ani = FuncAnimation(fig, update, frames=T, interval=20, blit=False)
    aniname = savefolder + "HD_activities.gif"
    ani.save(aniname, writer="Pillow", fps=30)