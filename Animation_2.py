import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
import jax
from Network_models import HD_cell, GD_cell_hexagonal, circle_period, GD_cell
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.signal import find_peaks, peak_widths

# simulation time step
bm.set_dt(1.)

# 环境设置
x1 = -np.pi
x2 = np.pi
x_bound = np.array([x1, x2, x2, x1, x1])

num_hd = 100
noise_stre = 0.1
v0 = bm.pi/1000 #baseline speed
offset_len = 1/7  * (1/v0) #v0/2
Animal_speed = 1.*v0 #vary this to change the speed of the animal

A = 3 #1.6
mbar_gc = 10. 
mabr_hd = 12.

alpha_hd = 0.4 
alpha_gc = 0.5
theta_modulation_stre_hd = alpha_hd * 1/v0 * Animal_speed
theta_modulation_stre_gc = alpha_gc * 1/v0 * Animal_speed

numT = int(4 * np.pi / (bm.dt * Animal_speed))

def Sweep_center(Ratio):
    bm.random.seed(0)
    Grid_net = GD_cell_hexagonal(
        ratio=Ratio,
        A=A,  # 3
        a=0.8,
        k=1.0,
        tau=10.,
        tau_v=100.,
        mbar=mbar_gc,  # 20.
        offset_len=offset_len,
        noise_stre=noise_stre,
        num_hd=num_hd,
    )

    HD_net = HD_cell(num=num_hd,
                     noise_stre=noise_stre / 100,
                     tau=10., tau_v=100.,
                     k=1., mbar=12, a=0.4, A=3., J0=4.,
                     z_min=-bm.pi, z_max=bm.pi)

    # run coupled net
    def run_CoupleNet(i, Animal_location, Head_direction, Moving_speed):  # 20 x size
        T_theta = 100
        # calculate ThetaModulator at time step i
        t = i * bm.dt
        theta_phase = bm.mod(t, T_theta) / T_theta  # theta phase（0,1）
        ThetaModulator_HD = 1 + theta_modulation_stre_hd * bm.cos(theta_phase * 2 * bm.pi)
        ThetaModulator_GC = 1 + theta_modulation_stre_gc * bm.cos(theta_phase * 2 * bm.pi)
        ThetaShutdown = 0.  # bm.where(theta_phase>0.5, -1, 0)

        # calculate internal head direction
        HD_net.step_run(i, Head_direction, ThetaModulator_HD)
        HD_activity = HD_net.r
        HD_center = HD_net.center

        # update the grid cell network
        Grid_net.step_run(i, Animal_location, HD_activity, ThetaModulator_GC, ThetaShutdown, Moving_speed, Head_direction)
        GC_bumpcenter = Grid_net.center_pos
        GC_bumpactivity = Grid_net.bump
        Animal_phase = Grid_net.Postophase(Animal_location)

        # get results
        GC_bumpcenter = Grid_net.center_pos
        Phase_bumpcenter = Grid_net.center_phase
        return GC_bumpcenter, GC_bumpactivity, HD_activity, HD_center, Phase_bumpcenter, Animal_phase

    # run network
    time_steps = np.arange(numT)

    def straight_line(x0, v, angle, T):
        x = []
        y = []
        xt = x0
        yt = x0
        for i in range(T):
            xt = xt + v * bm.cos(angle) * bm.dt
            yt = yt + v * bm.sin(angle) * bm.dt
            x.append(xt)
            y.append(yt)
        Animal_location = np.array([x, y])
        return Animal_location

    hd_angle = np.pi / 4
    Animal_location = straight_line(0, v=Animal_speed, angle=hd_angle, T=numT)
    Animal_location = Animal_location.transpose()
    Head_direction = hd_angle * bm.ones(numT)  # fixed head direction, mimicking the animal running in a straight line
    Moving_speed = Animal_speed * bm.ones([numT, 1])

    Grid_net.center_pos = Animal_location[0]
    center_grid, r_grid, HD_activity, HD_center, center_phase, Animal_phase = bm.for_loop(
        run_CoupleNet, (time_steps, Animal_location, Head_direction, Moving_speed), progress_bar=True
    )

    phi = Grid_net.value_bump
    return center_grid, HD_activity, r_grid, Animal_location, phi


Center = np.zeros([numT, 2, 3])
HD_activity = np.zeros([numT, 100, 3])
Grid_activity = np.zeros([numT, 10000, 3])
Ratio = np.array([2, 1, 0.5])
for i in range(3):
    center_grid, hd_r, r_grid, Animal_location, phi = Sweep_center(Ratio[i])
    Center[:, :, i] = center_grid
    HD_activity[:, :, i] = hd_r
    Grid_activity[:, :, i] = r_grid

onecycleT = numT/2
start = int(numT - 0.9*onecycleT)
end = int(numT - 0.15*onecycleT)

n_step = 10

fig, ax_ani = plt.subplots(2, 3, figsize=(10, 8), dpi=100)
hd_angle = np.pi/4
# Update first subplot to polar coordinates
ax_polar = plt.subplot(2, 3, 2, projection='polar')
R_hd = HD_activity[start:end, :, 1]
data_hd = R_hd[::n_step, :]
beta = np.linspace(-np.pi, np.pi, 100, endpoint=False)
ax_polar.plot([hd_angle, hd_angle], [0, np.max(R_hd[0, :])], 'k--')
ax_polar.set_ylim(-0.1, np.max(data_hd[0, :]) * 1.2)
line, = ax_polar.plot([], [])

ax_polar.set_title("direction sweep")

R_g = Grid_activity[start:end, :, :]
data = R_g[::n_step, :]
T = data.shape[0]

for i in range(3):
    ax_ani[1, i].set_aspect("equal")
    ax_ani[1, i].grid(True)

    ax_ani[1, i].plot(Animal_location[start:end, 0], Animal_location[start:end, 1], color="black")
    # ax_ani[1, i].plot(Center[:,0,i],Center[:,1,i], 'b')
    vmin1 = 0
    vmax1 = np.max(data[:, :, i])
    title = f"Grid Spacing: {np.pi * 2 / (2 ** (2 - i) / 2):.2f}"
    ax_ani[1, i].set_title(title)
    ax_ani[1, i].set_xlim(4, 10)
    ax_ani[1, i].set_ylim(4, 10)
    if i == 0:
        scatter1 = ax_ani[1, i].scatter([], [], c=[], s=200, cmap="Blues", vmin=vmin1, vmax=vmax1)
    if i == 1:
        scatter2 = ax_ani[1, i].scatter([], [], c=[], s=200, cmap="Blues", vmin=vmin1, vmax=vmax1)
    if i == 2:
        scatter3 = ax_ani[1, i].scatter([], [], c=[], s=200, cmap="Blues", vmin=vmin1, vmax=vmax1)

# 移除第一行的第一个和第三个子图
fig.delaxes(ax_ani[0, 0])
fig.delaxes(ax_ani[0, 2])

def update(frame):
    y1 = data[frame,:,0].flatten()
    y2 = data[frame,:,1].flatten()
    y3 = data[frame,:,2].flatten()
    y_hd = data_hd[frame].flatten()
    scatter1.set_offsets(np.column_stack((phi[:, 0], phi[:, 1])))
    scatter1.set_array(y1)
    scatter2.set_offsets(np.column_stack((phi[:, 0], phi[:, 1])))
    scatter2.set_array(y2)
    scatter3.set_offsets(np.column_stack((phi[:, 0], phi[:, 1])))
    scatter3.set_array(y3)
    line.set_data(beta, y_hd)
    return line, scatter1, scatter2, scatter3

ani = FuncAnimation(fig, update, frames=T, interval=100, blit=False)
plt.tight_layout()

aniname = "./Animation/Three_module_sweeps.gif"
ani.save(aniname, writer="Pillow", fps=10)
