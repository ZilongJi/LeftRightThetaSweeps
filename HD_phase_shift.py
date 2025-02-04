import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
import jax
from Network_models import HD_cell, circle_period
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from scipy.signal import find_peaks, peak_widths
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
theta_modulation_stre_hd = 0.4
mabr_hd = 12


HD_net = HD_cell(num=num_hd, 
                 noise_stre=noise_stre, 
                 tau=10., tau_v=100., 
                 k=1., mbar=mabr_hd, a=0.4, A=3., J0=4., 
                 z_min=-bm.pi, z_max=bm.pi)

# run coupled net
def run_CoupleNet(i, Head_direction):  # 20 x size

    T_theta = 100
    #calculate TheataModulator at time step i
    t = i*bm.dt
    theta_phase = bm.mod(t, T_theta)/T_theta # theta phase（0,1）
    ThetaModulator_HD = 1+theta_modulation_stre_hd*bm.cos(theta_phase*2*bm.pi)
    #calculate internal head direction
    HD_net.step_run(i, Head_direction, ThetaModulator_HD)
    Internal_direction = HD_net.center #center of mass of internal direction
    HD_activity = HD_net.r 

    return Internal_direction, HD_activity, theta_phase


# run network


N = 8
angular_velocity = 3.15/(1e3)
numT = 2000*20
time_steps = np.arange(numT)

def traj_hd(x0, v, T):
    x = []
    xt = x0
    for i in range(T):
        xt = xt + v * (1 + np.random.randn(1)) * bm.dt
        if xt > np.pi:
            xt -= 2 * np.pi
        if xt < -np.pi:
            xt += 2 * np.pi
        x.append(xt)
    return np.array(x)
Head_direction = traj_hd(x0=-np.pi, v=angular_velocity, T=numT)
plt.figure()
plt.plot(Head_direction)
plt.savefig('./HD_phase_shift/Head_direction.png', dpi=300)
center_HD, r_HD, theta_phase = bm.for_loop(
    run_CoupleNet, (time_steps, Head_direction), progress_bar=True
)
onecycleT = numT/(N/2)
start = int(numT - 0.7*onecycleT)
end = int(numT - 0.3*onecycleT)
theta_phase = (theta_phase-1/2)*2*np.pi
max_bump_activity = np.max(r_HD, axis=1)


labelsize = 10
ticksize = 8
s_size = 2

fig,ax = plt.subplots(1, 2, figsize=(6, 3.2), dpi=300)
hd_plot = Head_direction
hd_plot = np.stack([hd_plot, hd_plot]).flatten()
print(hd_plot.shape)
theta_phase_plot = theta_phase
theta_phase_plot = np.stack([theta_phase_plot, theta_phase_plot+np.pi*2]).flatten()
r_HD_plot = r_HD[:,50]
r_HD_plot = np.stack([r_HD_plot, r_HD_plot]).flatten()
sc = ax[0].scatter(
                hd_plot,
                theta_phase_plot,
                c=r_HD_plot,
                cmap="Blues",
                s=s_size,
)
ax[0].set_xlim(-1.5,1,5)
# ax[0].set_xlabel('Head direction')
ax[0].set_xlabel('Head direction')
ax[0].set_ylabel('Theta phase')
#activity colorbar
cbar = plt.colorbar(sc, ax=ax[0], shrink=0.7)
cbar.set_label('Firing rate of neuron #50')
#xlabels on cbar
ax[0].set_title('HD theta phase shift')

# ax[0].set_aspect('equal')
offset = center_HD[start:end].reshape(-1,)-Head_direction[start:end].reshape(-1,)
offset = circle_period(offset)
ax[1].plot(offset)
ax[1].set_xlabel('time')
ax[1].set_ylabel('Internal Head direction offset')
ax[1].set_title('Internal direction sweeps')
#xlabels on cbar


plt.tight_layout()

plt.savefig('./HD_phase_shift/HD_phase_shift.png', dpi=300)