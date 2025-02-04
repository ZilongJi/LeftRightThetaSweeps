import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
import jax
from Network_models import HD_cell, GD_cell_hexagonal, circle_period, GD_cell
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d

# simulation time step
bm.set_platform('gpu')
bm.set_dt(1.)

# 环境设置
x1 = -np.pi
x2 = np.pi
x_bound = np.array([x1, x2, x2, x1, x1])
y_bound = np.array([x1, x1, x2, x2, x1])

num_hd = 100
noise_stre = 0.5
v0 = bm.pi/1000
offset_len = 1/2 * (1/v0)
A = 3. #1.6
mbar = 20. 
Animal_speed = v0
alpha_gc = 0.5
alpha_hd = 0.3
theta_modulation_stre_hd = alpha_hd * 1/v0 * Animal_speed
theta_modulation_stre_gc = alpha_gc * 1/v0 * Animal_speed
Grid_net = GD_cell_hexagonal(
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
                 noise_stre=noise_stre/50, 
                 tau=10., tau_v=100., 
                 k=1., mbar=12, a=0.4, A=3., J0=4., 
                 z_min=-bm.pi, z_max=bm.pi)


def Speed_dependent_sweep_angle(Animal_speed, index, save):
    theta_modulation_stre_hd = alpha_hd * 1/v0 * Animal_speed
    theta_modulation_stre_gc = alpha_gc * 1/v0 * Animal_speed
    theta_modulation_stre_gc = bm.where(theta_modulation_stre_gc>0, theta_modulation_stre_gc, 0)


    numT = 8000
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
    Candidate_pos = Grid_net.Candidate_center
    Grid_net.center_pos = Animal_location[0]
    center_grid, center_HD, center_grid_input, r_grid, r_HD = bm.for_loop(
        run_CoupleNet, (time_steps, Animal_location, Head_direction, Moving_speed), progress_bar=True
    )

    max_r_grid = np.max(r_grid, axis=1) 
    start=int(numT/2)
    max_bump_activity = np.max(r_HD, axis=1)
    s_size = 2


    start=int(numT/2)
    data = circle_period(center_HD[start:].reshape(-1)-Head_direction[start:].reshape(-1))
    data = data/np.pi*180
    pks, _ = find_peaks(np.abs(data))
    distances = np.diff(pks)
    max_fr = gaussian_filter1d(max_bump_activity[start:], sigma=30)
    pks_hdfr, _ = find_peaks(max_fr)

    hd_direction = data[pks_hdfr]



    start=int(numT/2)
    dis_x_grid = center_grid[start:, 0].reshape(-1)-Animal_location[start:, 0].reshape(-1)
    dis_y_grid = center_grid[start:, 1].reshape(-1)-Animal_location[start:, 1].reshape(-1)
    dis_vec = bm.array([dis_x_grid, dis_y_grid])
    moving_vec = bm.array([bm.cos(hd_angle), bm.sin(hd_angle)]).reshape(1,2)
    projection = bm.matmul(moving_vec, dis_vec).reshape(-1,)


    # max_fr = gaussian_filter1d(projection, sigma=30)
    # pks_hdfr, _ = find_peaks(max_fr)
    data = gaussian_filter1d(projection, sigma=30)
    peaks_grid, _ = find_peaks(data)
    troughs_grid, _ = find_peaks(-data)
    # # 计算每个峰的宽度
    # widths_pks = peak_widths(data, peaks_grid, rel_height=0.5)[0]
    # widths_trs = peak_widths(-data, troughs_grid, rel_height=0.5)[0]
    # 筛选出具有最小宽度的峰
    # peaks_grid = peaks_grid[widths_pks >= 20]
    # troughs_grid = troughs_grid[widths_trs >= 20]
    # distances_peaks = np.diff(peaks_grid)
    # distances_troughs = np.diff(troughs_grid)

    W_rotate = np.array([[np.cos(np.pi/2-hd_angle), -np.sin(np.pi/2-hd_angle)], [np.sin(np.pi/2-hd_angle), np.cos(np.pi/2-hd_angle)]])
    Dis = np.array([dis_x_grid, dis_y_grid])
    Dis_rotate = np.matmul(W_rotate, Dis)
    max_r_grid = np.max(r_grid, axis=1) 
    
    Start = Dis_rotate[:, troughs_grid]
    End = Dis_rotate[:, peaks_grid]

    num_cycle_gc_troughs = troughs_grid.shape[0]
    num_cycle_gc_peaks = peaks_grid.shape[0]
    num_cycle_hd = hd_direction.shape[0]
    num_cycle = np.min(np.array([num_cycle_gc_troughs, num_cycle_gc_peaks, num_cycle_hd]))
    sweep_direction = np.zeros(num_cycle)
    # hd_direction = np.zeros(num_cycle)
    for i in range(num_cycle):
        sweep_direction[i] = np.arctan((End[0,i]-Start[0,i])/(End[1,i]-Start[1,i]))
    gc_angle = sweep_direction/np.pi*180
    if save == True:
        fig, ax = plt.subplots()
        sc = ax.scatter(
                        Dis_rotate[0,:],
                        Dis_rotate[1,:],
                        c=max_r_grid[start:],
                        cmap="cool",
                        s=10,
        )
        for i in range(num_cycle):
            sweep_vector = bm.array([bm.cos(sweep_direction[i]+np.pi/2),bm.sin(sweep_direction[i]+np.pi/2)])
            ax.arrow(0, 0, sweep_vector[0], sweep_vector[1], head_width=0.1, head_length=0.05, fc='red', ec='red')

            hd_sweep_angle = hd_direction[i]/180*np.pi
            hd_vector = bm.array([bm.cos(hd_sweep_angle+np.pi/2),bm.sin(hd_sweep_angle+np.pi/2)])
            # hd_vector = bm.matmul(W_rotate, hd_vector)
            ax.arrow(0, 0, hd_vector[0], hd_vector[1], head_width=0.1, head_length=0.05, fc='blue', ec='blue')
        plt.legend(['gc sweep', 'hd sweep'])

        # 添加 x=0 处的直线
        plt.axvline(x=0, color='k')
        # 添加 y=0 处的直线
        plt.axhline(y=0, color='k')
        ax.set_ylim(-1, 1.5)
        ax.set_xlim(-1.5, 1.5)
        filename = 'figure_sweep_angle/Sweep_direction_speed_' + str(index) + '.png'
        plt.savefig(filename)

    mean_gc_angle = np.mean(np.abs(gc_angle))
    mean_hd_direction = np.mean(np.abs(hd_direction))
    var_gc_angle = np.var(np.abs(gc_angle))
    var_hd_direction = np.var(np.abs(hd_direction))
    print('mean hd sweep angle:', mean_hd_direction)
    print('mean gc sweep angle:', mean_gc_angle)
    print('variance of hd sweep angle:', var_hd_direction)
    print('variance of gc sweep angle:', var_gc_angle)

    return mean_gc_angle, mean_hd_direction, var_gc_angle, var_hd_direction

num_speed = 20
Animal_speed = np.linspace(v0*0.3, v0*1.7, num_speed)
mean_gc_angle = np.zeros(num_speed,)
mean_hd_direction = np.zeros(num_speed,)
var_gc_angle = np.zeros(num_speed,)
var_hd_direction = np.zeros(num_speed,)
for ni in range(num_speed):
    print('Iteration:', ni)
    if ni % 3 == 0:
        save = True
    else:
        save = False
    mean_gc_angle[ni], mean_hd_direction[ni], var_gc_angle[ni], var_hd_direction[ni] = Speed_dependent_sweep_angle(Animal_speed=Animal_speed[ni], index = ni, save=save)


np.savez('sweeps.npz', 
         mean_gc_angle=mean_gc_angle, 
         mean_hd_direction=mean_hd_direction, 
         var_gc_angle=var_gc_angle, 
         var_hd_direction=var_hd_direction)


plt.figure()
plt.plot(Animal_speed*1000, mean_gc_angle)
plt.plot(Animal_speed*1000, mean_hd_direction)
plt.xlabel('Animal speed')
plt.ylabel('Sweep Angle')
plt.legend(['GC sweep angle', 'HD sweep angle'])
plt.plot(Animal_speed*1000, mean_gc_angle,'r.')
plt.plot(Animal_speed*1000, mean_gc_angle,'b.')
filename = 'figure_sweep_angle/Speed_dependent_angle.png'
plt.savefig(filename)


plt.figure()
plt.plot(Animal_speed*1000, var_gc_angle)
plt.plot(Animal_speed*1000, var_hd_direction)
plt.xlabel('Animal speed')
plt.ylabel('Sweep Angle Variance')
plt.legend(['GC sweep angle', 'HD sweep angle'])
plt.plot(Animal_speed*1000, var_gc_angle,'r.')
plt.plot(Animal_speed*1000, var_gc_angle,'b.')
filename = 'figure_sweep_angle/Speed_dependent_var.png'
plt.savefig(filename)
# plt.show()
