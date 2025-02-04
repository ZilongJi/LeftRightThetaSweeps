import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
import jax
from Network_models import HD_cell, GD_cell, traj
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from scipy.signal import find_peaks, peak_widths
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

bm.clear_buffer_memory()
bm.set_platform('gpu')
# simulation time step
bm.set_dt(1.)

# 环境设置
x1 = -np.pi
x2 = np.pi
x_bound = np.array([x1, x2, x2, x1, x1])
y_bound = np.array([x1, x1, x2, x2, x1])

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

Grid_net = GD_cell(
    A = A, # 3
    a = 0.8,
    k = 1.0,
    tau = 10.,
    tau_v = 100.,
    mbar = mbar_gc, # 20.
    offset_len = offset_len, 
    noise_stre = noise_stre,
    num_hd = num_hd,
)

HD_net = HD_cell(num=num_hd, 
                 noise_stre=noise_stre/100, 
                 tau=10., tau_v=100., 
                 k=1., mbar=mabr_hd, a=0.4, A=3., J0=4., 
                 z_min=-bm.pi, z_max=bm.pi)

# run coupled net
def run_CoupleNet(i, Animal_location, Head_direction, Moving_speed):  # 20 x size

    T_theta = 100
    #calculate TheataModulator at time step i
    t = i*bm.dt
    theta_phase = bm.mod(t, T_theta)/T_theta 
    ThetaModulator_HD = 1 + theta_modulation_stre_hd * bm.cos(theta_phase*2*bm.pi)
    ThetaModulator_GC = 1 + theta_modulation_stre_gc * bm.cos(theta_phase*2*bm.pi)
    
    #calculate internal head direction
    HD_net.step_run(i, Head_direction, ThetaModulator_HD)
    Internal_direction = HD_net.center #center of mass of internal direction
    HD_activity = HD_net.r 
    # print(HD_activity.shape)
          
    #update the grid cell network 
    Grid_net.step_run(i, Animal_location, HD_activity, ThetaModulator_GC, Moving_speed)
    GC_bumpcenter = Grid_net.center
    GC_bumpactivity = Grid_net.bump
    
    return GC_bumpcenter, Internal_direction, GC_bumpactivity, HD_activity


# run network

N = 8
numT = int(N*np.pi/(bm.dt*Animal_speed))
time_steps = np.arange(numT)

x = traj(x0=-np.pi, v=Animal_speed, T=numT)
Animal_location = bm.array([x, x]).transpose()
Head_direction = bm.pi/4*bm.ones(numT) #fixed head direction, mimicking the animal running in a straight line
Moving_speed = Animal_speed*bm.ones([numT,1])
#ThetaModulator = bm.ones(numT)+0.3*bm.sin(time_steps*2*bm.pi/100)
#ThetaShutdown = bm.zeros(numT)

center_grid, center_HD, r_grid, r_HD = bm.for_loop(
    run_CoupleNet, (time_steps, Animal_location, Head_direction, Moving_speed), progress_bar=True
)
onecycleT = numT/(N/2)
start = int(numT - 0.9*onecycleT)
end = int(numT - 0.15*onecycleT)

max_r_grid = np.max(r_grid, axis=1)
max_bump_activity = np.max(r_HD, axis=1)

hd_angle = np.pi/4
n_step = 10
phi = Grid_net.value_bump

fig = plt.figure(figsize=(10, 5), dpi=100)
gs = GridSpec(1, 2, width_ratios=[1, 1])

# 第一子图使用极坐标
ax_polar = fig.add_subplot(gs[0, 0], projection='polar')
R_hd = r_HD[start:end, :]
data_hd = R_hd[::n_step, :]
beta = np.linspace(-np.pi, np.pi, 100, endpoint=False)
ax_polar.plot([hd_angle, hd_angle], [0, np.max(R_hd[0, :])], 'k--')
ax_polar.set_ylim(-0.1, np.max(data_hd[0, :]) * 1.2)
line, = ax_polar.plot([], [])
ax_polar.set_title("direction sweep", pad=40)

# 第二子图使用笛卡尔坐标
ax_cartesian = fig.add_subplot(gs[0, 1])
R_g = r_grid[start:end, :]
data = R_g[::n_step, :]
T = data.shape[0]
ax_cartesian.set_aspect("equal")
ax_cartesian.grid(True)

ax_cartesian.plot(Animal_location[start:end, 0], Animal_location[start:end, 1], color="black")

vmin1 = 0
vmax1 = np.max(data)
scatter1 = ax_cartesian.scatter([], [], c=[], s=200, cmap="Blues", vmin=vmin1, vmax=vmax1)
ax_cartesian.set_title("location sweep", pad=40)

ax_cartesian.set_xlim(-3, 3)
ax_cartesian.set_ylim(-3, 3)

def update(frame):
    y1 = data[frame].flatten()
    y2 = data_hd[frame].flatten()
    scatter1.set_offsets(np.column_stack((phi[:, 0], phi[:, 1])))
    scatter1.set_array(y1)
    line.set_data(beta, y2)
    return line, scatter1

ani = FuncAnimation(fig, update, frames=T, interval=100, blit=False)
plt.tight_layout()

aniname = "./Animation/HD_grid_bump.gif"
ani.save(aniname, writer="Pillow", fps=10)
