import numpy as np
import brainpy.math as bm
import matplotlib.pyplot as plt
from Network_models import HD_cell, GD_cell_hexagonal, traj, circle_period
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.signal import find_peaks, peak_widths

# simulation time step
bm.set_dt(0.1)

# 环境设置
x1 = -np.pi
x2 = np.pi
x_bound = np.array([x1, x2, x2, x1, x1])
y_bound = np.array([x1, x1, x2, x2, x1])

num_hd = 100
noise_stre = 0.1
offset_len = 50/bm.pi
A = 3. #1.6
mbar = 20.
alpha_0 = 0.3
Grid_net = GD_cell_hexagonal(
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
                k=1., mbar=12, a=0.4, A=3., J0=4., 
                z_min=-bm.pi, z_max=bm.pi)

def Speed_dependent_sweeps(Animal_speed):
    theta_modulation_stre = alpha_0 * 200/bm.pi * (Animal_speed - bm.pi/200)
    numT = int(4*np.pi/(bm.dt*Animal_speed))
    # run coupled net
    def run_CoupleNet(i, Animal_location, Head_direction, Moving_speed):  # 20 x size
        T_theta = 10
        #calculate TheataModulator at time step i
        t = i*bm.dt
        theta_phase = bm.mod(t, T_theta)/T_theta # theta phase（0,1）
        ThetaModulator_HD = 1+0.3*bm.cos(theta_phase*2*bm.pi)
        ThetaModulator_GC = 1+theta_modulation_stre*bm.cos(theta_phase*2*bm.pi)
        ThetaShutdown=0.
        #calculate internal head direction
        HD_net.step_run(i, Head_direction, ThetaModulator_HD)
        HD_activity = HD_net.r 
        #update the grid cell network 
        Grid_net.step_run(i, Animal_location, HD_activity, ThetaModulator_GC, ThetaShutdown, Moving_speed)
        
        #get results
        GC_bumpcenter = Grid_net.center
        return GC_bumpcenter

    # run network
    time_steps = np.arange(numT)
    x = traj(x0=-np.pi, v=Animal_speed, T=numT)
    Animal_location = bm.array([x, x]).transpose()
    Head_direction = bm.pi/4*bm.ones(numT) #fixed head direction, mimicking the animal running in a straight line
    Moving_speed = Animal_speed*bm.ones([numT,1])
    center_grid = bm.for_loop(
        run_CoupleNet, (time_steps, Animal_location, Head_direction, Moving_speed), progress_bar=True
    )
    start=int(numT/2)
    dis_x_grid = circle_period(center_grid[start:, 0].reshape(-1)-Animal_location[start:, 0].reshape(-1))
    dis_y_grid = circle_period(center_grid[start:, 1].reshape(-1)-Animal_location[start:, 1].reshape(-1))
    dis_vec = bm.array([dis_x_grid, dis_y_grid])
    moving_vec = bm.array([bm.sqrt(2)/2, bm.sqrt(2)/2]).reshape(1,2)
    projection = bm.matmul(moving_vec, dis_vec).reshape(-1,)
    sweep_length = np.max(projection)-np.min(projection)
    # plt.plot(projection)
    print('sweep amplitude of grid cell sweeps', sweep_length)
    return sweep_length
    # sweep_amplitude = (np.max(data)-np.min(data))/2

num_speed = 10
Animal_speed = bm.linspace(bm.pi/100, bm.pi/100*2, num_speed)
sweep_length = bm.zeros(num_speed,)
for i in range(num_speed):
    sweep_length[i] = Speed_dependent_sweeps(Animal_speed[i])
    print('Iterative ', i)

sweep_length = bm.as_numpy(sweep_length)
# 保存数组为 .npy 文件
np.save('sweep_length.npy', sweep_length)

plt.plot(Animal_speed, sweep_length)
plt.show()