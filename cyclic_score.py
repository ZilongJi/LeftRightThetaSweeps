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
bm.set_dt(1.)

# 环境设置
x1 = -np.pi
x2 = np.pi
x_bound = np.array([x1, x2, x2, x1, x1])
y_bound = np.array([x1, x1, x2, x2, x1])

num_hd = 100
noise_stre = 0.01
v0 = bm.pi/1000
offset_len = 1/2 * (1/v0)
alpha_gc = 0.5
alpha_hd = 0.3



numT = 4000
def cyclic_score(A = 1.6, mbar=20, Animal_speed=v0/8, plot=True):
    theta_modulation_stre_hd = alpha_hd * (1/v0) * Animal_speed
    theta_modulation_stre_gc = alpha_gc * 1/v0 * Animal_speed
    theta_modulation_stre_gc = bm.where(theta_modulation_stre_gc>0, theta_modulation_stre_gc, 0)
    Grid_net = GD_cell_hexagonal(
        ratio = 1.,
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

    Grid_net.center_pos = Animal_location[0]
    center_grid, center_HD, center_grid_input, r_grid, r_HD = bm.for_loop(
        run_CoupleNet, (time_steps, Animal_location, Head_direction, Moving_speed), progress_bar=plot
    )

    start=int(numT/2)


    dis_x_grid = center_grid[start:, 0].reshape(-1)-Animal_location[start:, 0].reshape(-1)
    dis_y_grid = center_grid[start:, 1].reshape(-1)-Animal_location[start:, 1].reshape(-1)

    data = np.sqrt(dis_x_grid**2 + dis_y_grid**2)
    peaks_grid, _ = find_peaks(data)
    troughs_grid, _ = find_peaks(-data)
    # 计算每个峰的宽度
    widths_pks = peak_widths(data, peaks_grid, rel_height=0.5)[0]
    widths_trs = peak_widths(-data, troughs_grid, rel_height=0.5)[0]
    # 筛选出具有最小宽度的峰
    peaks_grid = peaks_grid[widths_pks >= 20]
    troughs_grid = troughs_grid[widths_trs >= 20]

    Dis = np.array([dis_x_grid, dis_y_grid])
    Start = Dis[:, troughs_grid]
    End = Dis[:, peaks_grid]

    num_cycle_gc_troughs = troughs_grid.shape[0]
    num_cycle_gc_peaks = peaks_grid.shape[0]

    num_cycle = np.min(np.array([num_cycle_gc_troughs, num_cycle_gc_peaks]))

    sweep_direction = np.zeros(num_cycle)
    long_axis_projection = np.zeros(int(numT/2)-troughs_grid[0])
    short_axis_projection = np.zeros(int(numT/2)-troughs_grid[0])

    for i in range(num_cycle-1):
        sweep_direction[i] = np.arctan((End[1,i]-Start[1,i])/(End[0,i]-Start[0,i]))
        sweep_vector = np.array([np.cos(sweep_direction[i]),np.sin(sweep_direction[i])])
        sweep_vector_perp = np.array([np.cos(sweep_direction[i]-np.pi/2),np.sin(sweep_direction[i]-np.pi/2)])
        if i >= 1:
            phase_window = troughs_grid[i] - troughs_grid[i-1]
            for j in range(phase_window):
                long_axis_projection[troughs_grid[i-1]+j-troughs_grid[0]] = np.sum(Dis[:,troughs_grid[i]+j].reshape(-1,)*sweep_vector.reshape(-1,))
                short_axis_projection[troughs_grid[i-1]+j-troughs_grid[0]] = np.sum(Dis[:,troughs_grid[i]+j].reshape(-1,)*sweep_vector_perp.reshape(-1,))


    long_axis = np.max(long_axis_projection)-np.min(long_axis_projection)
    short_axis = np.max(short_axis_projection)-np.min(short_axis_projection)
    cyclic_score = short_axis/long_axis
    if cyclic_score>1:
        cyclic_score = 1/cyclic_score
    if long_axis<0.3:
        cyclic_score=0
    print(cyclic_score)
    print(short_axis)
    print(long_axis)
    
    def plot_data():
        Dis = np.array([dis_x_grid, dis_y_grid])
        max_r_grid = np.max(r_grid, axis=1) 
        fig, ax = plt.subplots()
        sc = ax.scatter(
                        Dis[0,:],
                        Dis[1,:],
                        c=max_r_grid[start:],
                        cmap="cool",
                        s=10,
        )
        Start = Dis[:, troughs_grid]
        End = Dis[:, peaks_grid]

        num_cycle_gc_troughs = troughs_grid.shape[0]
        num_cycle_gc_peaks = peaks_grid.shape[0]
        num_cycle = np.min(np.array([num_cycle_gc_troughs, num_cycle_gc_peaks]))

        # num_cycle = troughs_grid.shape[0]
        sweep_direction = np.zeros(num_cycle)
        long_axis_projection = np.zeros(int(numT/2)-troughs_grid[0])
        short_axis_projection = np.zeros(int(numT/2)-troughs_grid[0])

        for i in range(num_cycle-1):
            sweep_direction[i] = np.arctan((End[1,i]-Start[1,i])/(End[0,i]-Start[0,i]))
            sweep_vector = np.array([np.cos(sweep_direction[i]),np.sin(sweep_direction[i])])
            sweep_vector_perp = np.array([np.cos(sweep_direction[i]-np.pi/2),np.sin(sweep_direction[i]-np.pi/2)])

            ax.arrow(0, 0, sweep_vector[0], sweep_vector[1], head_width=0.1, head_length=0.05, fc='red', ec='red')
            ax.arrow(0, 0, sweep_vector_perp[0], sweep_vector_perp[1], head_width=0.1, head_length=0.05, fc='red', ec='blue')
            if i >= 1:
                phase_window = troughs_grid[i] - troughs_grid[i-1]
                for j in range(phase_window):
                    long_axis_projection[troughs_grid[i-1]+j-troughs_grid[0]] = np.sum(Dis[:,troughs_grid[i]+j].reshape(-1,)*sweep_vector.reshape(-1,))
                    short_axis_projection[troughs_grid[i-1]+j-troughs_grid[0]] = np.sum(Dis[:,troughs_grid[i]+j].reshape(-1,)*sweep_vector_perp.reshape(-1,))

        plt.legend(['gc sweep', 'hd sweep'])
        plt.plot(Start[0,:], Start[1,:],'r.')
        plt.plot(End[0,:], End[1,:],'r.')

        # 添加 x=0 处的直线
        plt.axvline(x=0, color='k')
        # 添加 y=0 处的直线
        plt.axhline(y=0, color='k')
        ax.set_ylim(-1, 4)
        ax.set_xlim(-np.pi, np.pi)
        plt.savefig('cyclic_pattern.png')

        plt.figure()
        plt.plot(long_axis_projection)
        plt.plot(short_axis_projection)
        plt.savefig('axis_projection.png')

    if plot == True:
        plot_data()

    return cyclic_score


num_A = 10
input_A = np.linspace(1,2.6,num_A)
Cyclic_score = np.zeros(num_A)
for i in range(num_A):
    print('Iterative', i)
    Cyclic_score[i] = cyclic_score(A=input_A[i], plot=False)
plt.figure()
plt.plot(input_A, Cyclic_score)
plt.plot(input_A, Cyclic_score,'r.')
plt.xlabel('Conj-Grid input strength')
plt.ylabel('Cyclic_score')
plt.savefig('Cyclic_score.png')

# cyclic_score(A=2.6, plot=True)