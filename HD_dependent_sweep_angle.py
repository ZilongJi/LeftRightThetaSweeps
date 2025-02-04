import numpy as np
import brainpy.math as bm
import matplotlib.pyplot as plt
from Network_models import HD_cell, GD_cell, traj, circle_period
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths


# simulation time step
bm.set_dt(0.1)

# 环境设置
x1 = -np.pi
x2 = np.pi
x_bound = np.array([x1, x2, x2, x1, x1])
y_bound = np.array([x1, x1, x2, x2, x1])

num_hd = 100
noise_stre = 0.5
offset_len = 50/bm.pi
Animal_speed = bm.pi/100
A = 3. #1.6
mbar = 20.
alpha_0 = 0.3
theta_modulation_stre = 0.3


numT = int(4*np.pi/(bm.dt*Animal_speed))
def hd_dependent_sweep_angle(A_hd):
    Grid_net = GD_cell(
        A = A, # 3
        a = 0.8,
        k = 1.0,
        tau = 1.,
        tau_v = 10.,
        mbar = mbar, # 20.
        offset_len = offset_len, 
        noise_stre = noise_stre,
        num_hd = num_hd,
    )
    HD_net = HD_cell(num=num_hd, 
                    noise_stre=0.01, 
                    tau=1., tau_v=10., 
                    k=1., mbar=12, a=0.4, A=A_hd, J0=4., 
                    z_min=-bm.pi, z_max=bm.pi)

    # run coupled net
    def run_CoupleNet(i, Animal_location, Head_direction, Moving_speed):  # 20 x size
        manual_HD = False
        if not manual_HD:
            T_theta = 10
            #calculate TheataModulator at time step i
            t = i*bm.dt
            theta_phase = bm.mod(t, T_theta)/T_theta # theta phase（0,1）
            ThetaModulator_HD = 1+0.3*bm.cos(theta_phase*2*bm.pi)
            ThetaModulator_GC = 1+theta_modulation_stre*bm.cos(theta_phase*2*bm.pi)
            #calculate ThetaShutdown at time step i (whether to shut down the newtork activity or not)
            ThetaShutdown = 0. #bm.where(theta_phase>0.5, -1, 0)
            
            #calculate internal head direction
            HD_net.step_run(i, Head_direction, ThetaModulator_HD)
            Internal_direction = HD_net.center #center of mass of internal direction
            HD_activity = HD_net.r 
            
        else:
            T_theta = 20
            t = i*bm.dt
            n_cycle = t//T_theta #第几个theta cycle
            theta_phase = bm.mod(t, T_theta)/T_theta # theta phase（0,1）
            Internal_direction = 0.5* theta_phase * bm.where(bm.mod(n_cycle,2)==0, 1, -1) + bm.pi/4#偶数个theta cycle逆时针旋转，奇数顺时针
            ThetaModulator_GC = bm.exp(-theta_phase*4)*4 #1.0
            # theta_input = bm.sign(bm.sin(t*2*bm.pi/T_theta)) # Theta input is moved out from Grid cell class
            ThetaShutdown = bm.where(theta_phase>0.8, -10, 0) #artificially shut down the bump activities    
            
        #update the grid cell network 
        Grid_net.step_run(i, Animal_location, HD_activity, ThetaModulator_GC, ThetaShutdown, Moving_speed)
        
        #get results
        GC_bumpcenter = Grid_net.center
        center_grid_input = Grid_net.center_conjG
        GC_bumpactivity = Grid_net.r
        return GC_bumpcenter, Internal_direction, center_grid_input, GC_bumpactivity, HD_activity


    # run network

    time_steps = np.arange(numT)

    x = traj(x0=-np.pi, v=Animal_speed, T=numT)
    Animal_location = bm.array([x, x]).transpose()
    Head_direction = bm.pi/4*bm.ones(numT) #fixed head direction, mimicking the animal running in a straight line
    Moving_speed = Animal_speed*bm.ones([numT,1])
    #ThetaModulator = bm.ones(numT)+0.3*bm.sin(time_steps*2*bm.pi/100)
    #ThetaShutdown = bm.zeros(numT)

    center_grid, center_HD, center_grid_input, r_grid, r_HD = bm.for_loop(
        run_CoupleNet, (time_steps, Animal_location, Head_direction, Moving_speed), progress_bar=False
    )


    start=int(numT/2)
    data = circle_period(center_HD[start:].reshape(-1)-Head_direction[start:].reshape(-1))
    peaks_hd, _ = find_peaks(data)
    distances = np.diff(peaks_hd)
    sweep_amplitude = (np.max(data)-np.min(data))/2
    data = circle_period(center_HD[start:].reshape(-1)-Head_direction[start:].reshape(-1))
    data = data/np.pi*180
    pks, _ = find_peaks(np.abs(data))
    distances = np.diff(pks)
    max_bump_activity = np.max(r_HD, axis=1)
    max_fr = gaussian_filter1d(max_bump_activity[start:], sigma=30)
    pks_hdfr, _ = find_peaks(max_fr)

    hd_direction = data[pks_hdfr]


    start=int(numT/2)
    dis_x_grid = circle_period(center_grid[start:, 0].reshape(-1)-Animal_location[start:, 0].reshape(-1))
    dis_y_grid = circle_period(center_grid[start:, 1].reshape(-1)-Animal_location[start:, 1].reshape(-1))
    dis_vec = bm.array([dis_x_grid, dis_y_grid])
    moving_vec = bm.array([bm.sqrt(2)/2, bm.sqrt(2)/2]).reshape(1,2)
    projection = bm.matmul(moving_vec, dis_vec).reshape(-1,)


    data = projection
    peaks_grid, _ = find_peaks(data)
    troughs_grid, _ = find_peaks(-data)
    # 计算每个峰的宽度
    widths_pks = peak_widths(data, peaks_grid, rel_height=0.5)[0]
    widths_trs = peak_widths(-data, troughs_grid, rel_height=0.5)[0]
    # 筛选出具有最小宽度的峰
    peaks_grid = peaks_grid[widths_pks >= 20]
    troughs_grid = troughs_grid[widths_trs >= 20]

    distances_peaks = np.diff(peaks_grid)
    distances_troughs = np.diff(troughs_grid)

    W_rotate = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
    Dis = np.array([dis_x_grid, dis_y_grid])
    Dis_rotate = np.matmul(W_rotate, Dis)
    max_r_grid = np.max(r_grid, axis=1) 

    Start = Dis_rotate[:, troughs_grid]
    End = Dis_rotate[:, peaks_grid]

    num_cycle = troughs_grid.shape[0]
    sweep_direction = np.zeros(num_cycle)
    for i in range(num_cycle):
        sweep_direction[i] = np.arctan((End[0,i]-Start[0,i])/(End[1,i]-Start[1,i]))/np.pi*180

    hd_sweep_angle = np.mean(np.abs(hd_direction))
    gc_sweep_angle = np.mean(np.abs(sweep_direction))

    return hd_sweep_angle, gc_sweep_angle


A_hd = np.linspace(2.5, 4, 10)
hd_sweep_angle = np.zeros(10,)
gc_sweep_angle = np.zeros(10,)

for i in range(10):
    hd_sweep_angle[i], gc_sweep_angle[i] = hd_dependent_sweep_angle(A_hd[i])
    print('Iterative', i)

plt.plot(hd_sweep_angle, gc_sweep_angle)
plt.xlabel('hd sweep angle')
plt.ylabel('gc_sweep_angle')
plt.savefig('hd_dependent_sweep_angle.png')



