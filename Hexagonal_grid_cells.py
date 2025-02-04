import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
import jax
from Network_models import HD_cell, GD_cell, traj, circle_period
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.signal import find_peaks, peak_widths
# Grid cell model modules
class GD_cell_hexagonal(bp.DynamicalSystem):
    def __init__(
        self,
        ratio=1,
        noise_stre=0.01,
        num=100,
        tau=1.0,
        tau_v=10.0,
        mbar=75.0,
        a=0.5,
        A=1.0,
        J0=5.0,
        k=1,
        g = 1000,
        x_min=-bm.pi,
        x_max=bm.pi,
        offset_len=0.0,
        num_hd = 100,
    ):
        super(GD_cell_hexagonal, self).__init__()

        # dynamics parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v  # The time constant of the adaptation variable
        self.ratio = ratio
        self.num_x = num  # number of excitatory neurons for x dimension
        self.num_y = num  # number of excitatory neurons for y dimension
        self.num = self.num_x * self.num_y
        self.num_hd = num_hd
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.g = g
        self.J0 = J0/g  # maximum connection value
        self.m = mbar * tau / tau_v
        self.noise_stre = noise_stre
        self.offset_len = offset_len
        self.Lambda = bm.pi*2/self.ratio

        # feature space
        self.x_range = x_max - x_min
        phi_x = bm.linspace(x_min, x_max, self.num_x + 1)  # The encoded feature values
        self.x = phi_x[0:-1]
        self.y_range = self.x_range
        phi_y = bm.linspace(x_min, x_max, self.num_y + 1)  # The encoded feature values
        self.y = phi_y[0:-1]
        x_grid, y_grid = bm.meshgrid(self.x, self.y)
        self.x_grid = x_grid.flatten()
        self.y_grid = y_grid.flatten()
        self.value_grid = bm.stack([self.x_grid, self.y_grid]).T
        self.rho = self.num / (self.x_range * self.y_range)  # The neural density
        self.dxy = 1 / self.rho  # The stimulus density
        self.coor_transform = bm.array([[1 , -1/bm.sqrt(3)],[0, 2/bm.sqrt(3)]])
        self.coor_transform_inv = np.linalg.inv(self.coor_transform)
        self.pos_grid = bm.matmul(self.coor_transform_inv, bm.transpose(self.value_grid)).T

        N_c = 5
        Candidate_center = bm.zeros([N_c,N_c,2])

        for ni in range(N_c):
            for nj in range(N_c):
                Candidate_center[ni,nj,0] = ni*self.Lambda - nj*self.Lambda/bm.sqrt(3)
                Candidate_center[ni,nj,1] = nj*self.Lambda * 2/bm.sqrt(3)
        self.Candidate_center = Candidate_center.reshape([N_c**2,2])

        # initialize conn matrix
        self.conn_mat = self.make_conn()

        # initialize dynamical variables
        self.r = bm.Variable(bm.zeros(self.num))
        self.u = bm.Variable(bm.zeros(self.num))
        self.v = bm.Variable(bm.zeros(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.center_phase = bm.Variable(bm.zeros(2))
        self.center_pos = bm.Variable(bm.zeros(2))
        self.center_I = bm.Variable(bm.zeros(2))

        # 定义积分器
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    def circle_period(self, d):
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        d = bm.where(d < -bm.pi, d + 2 * bm.pi, d)
        return d

    def dist(self, d):
        d = self.circle_period(d)
        dis = bm.matmul(self.coor_transform_inv, bm.transpose(d)).T
        delta_x = dis[:, 0]
        delta_y = dis[:, 1]
        dis = bm.sqrt(delta_x ** 2 + delta_y ** 2)
        return dis

    def make_conn(self):
        @jax.vmap
        def get_J(v):
            d = self.dist(v - self.value_grid)
            Jxx = (
                self.J0
                * bm.exp(-0.5 * bm.square(d / self.a))
                / (bm.sqrt(2 * bm.pi) * self.a)
            )
            return Jxx

        return get_J(self.value_grid)

    def Postophase(self, pos):
        Loc = pos * self.ratio  # ratio = inverse of grid scale
        phase = bm.matmul(self.coor_transform, Loc) + bm.pi  # 坐标变换
        phase_x = bm.mod(phase[0], 2 * bm.pi) - bm.pi
        phase_y = bm.mod(phase[1], 2 * bm.pi) - bm.pi
        Phase = bm.array([phase_x, phase_y])
        return Phase

    def input_by_conjG(self, pos, hd, input_stre):
        assert bm.size(pos) == 2
        offset = bm.array([bm.cos(hd), bm.sin(hd)]) * self.offset_len
        self.center_conjG = self.Postophase(
            pos.reshape(-1,) + offset.reshape(-1,)
        )  # Ideal phase using mapping function
        d = self.dist(bm.asarray(self.center_conjG) - self.value_grid)
        return self.A * bm.exp(-0.25 * bm.square(d / self.a)) * input_stre
    

    def input_by_conjG_new(self, Animal_location, HD_activity, ThetaModulator, Animal_speed):
        assert bm.size(Animal_location) == 2
        num_hd = self.num_hd
        hd = bm.linspace(0,bm.pi*2,num_hd) 
        # each head-direction cell corresponds to a group of Conjunctive grid cells, which in turn projects to pure grid cells with assymetric connections determined by offset(hd)
        offset = bm.array([bm.cos(hd), bm.sin(hd)]) * self.offset_len * Animal_speed
        self.center_conjG = self.Postophase(
            Animal_location.reshape([-1,1]) + offset.reshape(-1,num_hd)
        )  # Ideal phase using mapping function
        input = bm.zeros([num_hd, self.num])
        for i in range(num_hd):
            d = self.dist(bm.asarray(self.center_conjG[:,i]) - self.value_grid)
            input[i] = self.A * bm.exp(-0.25 * bm.square(d / self.a))
        hd_weight = bm.softmax(HD_activity) #not necessary
        # hd_weight = HD_activity
        hd_weight = hd_weight/bm.sum(hd_weight)
        # hd_weight = bm.zeros([num_hd])
        # bumpcenter = bm.argmax(HD_activity)
        # hd_weight[bumpcenter] = 1

        total_input = bm.matmul(input.transpose(), hd_weight).reshape(-1,) * ThetaModulator
        return total_input


    def get_center(self, pos):
        exppos_x = bm.exp(1j * self.x_grid)
        exppos_y = bm.exp(1j * self.y_grid)
        r = bm.where(self.r > bm.max(self.r) * 0.1, self.r, 0)
        center_phase = bm.zeros(2,)
        center_phase[0] = bm.angle(bm.sum(exppos_x * r))
        center_phase[1] = bm.angle(bm.sum(exppos_y * r))
        center_pos = bm.zeros(2,)
        center_pos = bm.matmul(self.coor_transform_inv, center_phase)
        # print(center_pos.shape)
        
        Candidate_center = self.Candidate_center + center_pos
        distances = bm.linalg.norm(Candidate_center - pos, axis=1)

        # 找到最小距离的点的索引
        closest_index = bm.argmin(distances)

        # 找到最近的点
        self.center_pos = Candidate_center[closest_index]
        self.center_phase.value = center_phase

    @property
    def derivative(self):
        du = (
            lambda u, t, Irec: (
                -u
                + Irec
                + self.input
                - self.v
            )
            / self.tau
        )
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def reset_state(self):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.v.value = bm.Variable(bm.zeros(self.num))
        self.input.value = bm.Variable(bm.zeros(self.num))

    def update(self, Animal_location, HD_activity, ThetaModulator, ThetaShutdown, speed):
        self.get_center(Animal_location)
        # input_conjG = self.input_by_conjG(pos, hd, input_stre)
        input_conjG = self.input_by_conjG_new(Animal_location, HD_activity, ThetaModulator, speed)
        
        self.input = input_conjG + ThetaShutdown
        
        Irec = bm.matmul(self.conn_mat, self.r) + self.noise_stre * bm.random.randn(
            (self.num)
        )
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share["t"], Irec, bm.dt)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = self.g*r1 / r2


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
Animal_speed = bm.pi/100 
A = 3. #1.6
mbar = 20.
alpha_0 = 0.3
theta_modulation_stre = alpha_0 * 200/bm.pi * (Animal_speed - bm.pi/200)


numT = int(4*np.pi/(bm.dt*Animal_speed))
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

# run coupled net
def run_CoupleNet(i, Animal_location, Head_direction, Moving_speed):  # 20 x size
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
    #update the grid cell network 
    Grid_net.step_run(i, Animal_location, HD_activity, ThetaModulator_GC, ThetaShutdown, Moving_speed)
    
    #get results
    GC_bumpcenter = Grid_net.center_pos
    center_grid_input = Grid_net.center_conjG
    GC_bumpactivity = Grid_net.r
    return GC_bumpcenter, Internal_direction, center_grid_input, GC_bumpactivity, HD_activity


# run network

time_steps = np.arange(numT)

def traj_new(x0, v, T):
    x = []
    xt = x0
    for i in range(T):
        xt = xt + v * bm.dt
        x.append(xt)
    return np.array(x)

x = traj_new(0, v=Animal_speed, T=numT)


Animal_location = bm.array([x, x]).transpose()
Head_direction = bm.pi/4*bm.ones(numT) #fixed head direction, mimicking the animal running in a straight line
Moving_speed = Animal_speed*bm.ones([numT,1])
#ThetaModulator = bm.ones(numT)+0.3*bm.sin(time_steps*2*bm.pi/100)
#ThetaShutdown = bm.zeros(numT)



center_grid, center_HD, center_grid_input, r_grid, r_HD = bm.for_loop(
    run_CoupleNet, (time_steps, Animal_location, Head_direction, Moving_speed), progress_bar=True
)
pos_grid = Grid_net.pos_grid

max_r_grid = np.max(r_grid, axis=1) 
start=int(numT/2)
max_bump_activity = np.max(r_HD, axis=1)
fig, axs = plt.subplots(2, 2, figsize=(6, 4.5))
s_size = 2

ax = axs[0, 0]
ax.plot(time_steps[start:], Head_direction[start:], linewidth=1, color='black')

# cb = ax.scatter(time_steps[10:], 
#                 center_HD[10:], 
#                 c=max_bump_activity[10:], 
#                 cmap='cool', s=s_size)
cb = ax.scatter(time_steps[start:], 
                center_HD[start:], 
                c=max_bump_activity[start:], 
                cmap='cool', s=s_size)
ax.set_ylim(-bm.pi, bm.pi)
#colorbar
cbar = plt.colorbar(cb, ax=ax, shrink=0.8)
ax.set_title('Internal direction')

ax = axs[0, 1]
sc = ax.scatter(
                center_grid[start:, 0],
                center_grid[start:, 1],
                c=max_r_grid[start:],
                cmap="cool",
                s=s_size,
)
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
ax.plot(Animal_location[start:, 0], Animal_location[start:, 1], color="black")
ax.set_title('GC bump')


ax = axs[1, 0]
# ax.scatter(range(len(center_grid[start:, 0])), x[start:], s=1, color='grey')
ax.plot(time_steps[start:], x[start:], color='grey', linewidth=1)
ax.set_ylabel("x")
sc = ax.scatter(
                time_steps[start:],
                center_grid[start:, 0],
                c=max_r_grid[start:],
                cmap="cool",
                s=s_size,
)
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
ax.set_title('bump x offset')


ax = axs[1, 1]
# ax.scatter(range(len(center_grid[start:, 0])), x[start:], s=1, color='grey')
ax.plot(time_steps[start:], x[start:], color='grey', linewidth=1)
ax.set_ylabel("y")
sc = ax.scatter(
                time_steps[start:],
                center_grid[start:, 1],
                c=max_r_grid[start:],
                cmap="cool",
                s=s_size,
)
cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
ax.set_title('bump y offset')

plt.tight_layout()
plt.show()