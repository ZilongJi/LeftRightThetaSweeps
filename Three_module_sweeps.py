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
y_bound = np.array([x1, x1, x2, x2, x1])

num_hd = 100
noise_stre = 0.2
v0 = bm.pi/1000
offset_len = 1/2 * (1/v0)
Animal_speed = v0
A = 3. #1.6
mbar = 20. 
alpha_gc = 0.5
alpha_hd = 0.3
# theta_modulation_stre_hd = alpha_hd * (1/v0) * Animal_speed
# theta_modulation_stre_hd = alpha_hd * 1/v0 * Animal_speed
theta_modulation_stre_hd = alpha_hd * (1/v0) * Animal_speed
theta_modulation_stre_gc = alpha_gc * 1/v0 * Animal_speed
theta_modulation_stre_gc = bm.where(theta_modulation_stre_gc>0, theta_modulation_stre_gc, 0)


numT = 4000



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
    Animal_location = np.array([x,y])
    return Animal_location

hd_angle = np.pi/6
Animal_location = straight_line(0, v=Animal_speed, angle = hd_angle, T=numT)
Animal_location = Animal_location.transpose()
Head_direction = hd_angle*bm.ones(numT) #fixed head direction, mimicking the animal running in a straight line
Moving_speed = Animal_speed*bm.ones([numT,1])

Ratio = np.array([2,1,0.5])
fig, axs = plt.subplots(1, 3, figsize=(9, 2.25))
for i in range(3):
    Grid_net = GD_cell_hexagonal(
        ratio = Ratio[i],
        A = A, # 3
        a = 0.8,
        k = 1.0,
        tau = 10.,
        tau_v = 100.,
        mbar = mbar, # 20.
        offset_len = offset_len, 
        noise_stre = noise_stre,
        num_hd = num_hd,
    )

    HD_net = HD_cell(num=num_hd, 
                    noise_stre=noise_stre/100, 
                    tau=10., tau_v=100., 
                    k=1., mbar=20, a=0.4, A=3., J0=4., 
                    z_min=-bm.pi, z_max=bm.pi)

    # run coupled net
    def run_CoupleNet(i, Animal_location, Head_direction, Moving_speed):  # 20 x size
        T_theta = 100
        #calculate TheataModulator at time step i
        t = i*bm.dt
        theta_phase = bm.mod(t, T_theta)/T_theta # theta phase（0,1）
        ThetaModulator_HD = 1+theta_modulation_stre_hd*bm.cos(theta_phase*2*bm.pi)
        ThetaModulator_GC = 1+theta_modulation_stre_gc*bm.cos(theta_phase*2*bm.pi)
        #calculate ThetaShutdown at time step i (whether to shut down the newtork activity or not)
        ThetaShutdown = 0. #bm.where(theta_phase>0.5, -1, 0)
        
        #calculate internal head direction
        HD_net.step_run(i, Head_direction, ThetaModulator_HD)
        Internal_direction = HD_net.center #center of mass of internal direction
        HD_activity = HD_net.r 
        #update the grid cell network 
        Grid_net.step_run(i, Animal_location, HD_activity, ThetaModulator_GC, ThetaShutdown, Moving_speed)
        
        #get results
        GC_bumpcenter = Grid_net.center_pos
        center_grid_input = Grid_net.center_conjG
        GC_bumpactivity = Grid_net.r
        return GC_bumpcenter, Internal_direction, center_grid_input, GC_bumpactivity, HD_activity

    center_grid, center_HD, center_grid_input, r_grid, r_HD = bm.for_loop(
        run_CoupleNet, (time_steps, Animal_location, Head_direction, Moving_speed), progress_bar=True
    )

    max_r_grid = np.max(r_grid, axis=1) 
    start=int(numT/2)

    s_size = 2

    ax = axs[i]
    sc = ax.scatter(
                    center_grid[start:, 0],
                    center_grid[start:, 1],
                    c=max_r_grid[start:],
                    cmap="cool",
                    s=s_size,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    ax.plot(Animal_location[start:, 0], Animal_location[start:, 1], color="black")
    Spacing = Grid_net.Lambda/np.pi
    title = 'Grid Spacing='+str(Spacing)
    ax.set_title(title)


plt.tight_layout()
filename = 'Three_module_sweeps.png'
plt.savefig(filename)