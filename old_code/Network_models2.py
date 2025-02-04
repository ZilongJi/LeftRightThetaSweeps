import jax
import brainpy as bp
import brainpy.math as bm
import numpy as np
import os

# tau=0.01, tauv=0.2, m0=1., k=1, a=1.0, A=4., J0=4.


class HD_cell(bp.DynamicalSystem):
    def __init__(
        self,
        num,
        noise_stre=0.01,
        tau=1.0,
        tau_v=10.0,
        k=0.8,
        mbar=2,
        a=bm.pi / 4,
        A=3.0,
        J0=5.0,
        z_min=-bm.pi,
        z_max=bm.pi,
    ):
        super(HD_cell, self).__init__()

        # parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v
        self.k = k / num * 20  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0 / num * 20  # maximum connection value
        self.m = mbar * tau / tau_v
        self.noise_stre = noise_stre

        # neuron num
        self.num = num  # head-direction cell
        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        x1 = bm.linspace(z_min, z_max, num + 1)  # The encoded feature values
        self.x = x1[0:-1]
        self.rho = num / self.z_range  # The neural density
        self.dx = self.z_range / num  # The stimulus density

        # connection matrix
        conn_mat = self.make_conn()
        self.conn_fft = bm.fft.fft(conn_mat)

        # neuron state variables
        self.r = bm.Variable(bm.zeros(num))
        self.u = bm.Variable(bm.zeros(num))  # head direction cell
        self.v = bm.Variable(bm.zeros(num))
        self.input = bm.Variable(bm.zeros(num))
        self.center = bm.Variable(bm.zeros(1))
        self.centerI = bm.Variable(bm.zeros(1))

        # 定义积分器
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    def dist(self, d):
        d = self.circle_period(d)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self):
        d = self.dist(bm.abs(self.x[0] - self.x))
        Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (2 * bm.pi * self.a ** 2)
        return Jxx

    def get_center(self, r, x):
        exppos = bm.exp(1j * x)
        center = bm.angle(bm.sum(exppos * r))
        return center.reshape(-1,)

    @property
    def derivative(self):
        du = lambda u, t, input: (-u + input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def circle_period(self, A):
        B = bm.where(A > bm.pi, A - 2 * bm.pi, A)
        B = bm.where(B < -bm.pi, B + 2 * bm.pi, B)
        return B

    def input_HD(self, HD):
        # integrate self motion
        return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - HD) / self.a))

    def reset_state(self, HD_truth):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.v.value = bm.Variable(bm.zeros(self.num))
        self.center.value = bm.Variable(bm.zeros(1,) + HD_truth)

    def update(self, HD, ThetaInput):
        self.center = self.get_center(r=self.r, x=self.x)
        Iext = ThetaInput * self.input_HD(HD)
        # Calculate input
        r_fft = bm.fft.fft(self.r)
        Irec = bm.real(bm.fft.ifft(r_fft * self.conn_fft))
        input_total = Iext + Irec + bm.random.randn(self.num) * self.noise_stre
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share.load("t"), input_total, bm.dt)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2


# Grid cell model modules
class GD_cell(bp.DynamicalSystem):
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
        x_min=-bm.pi,
        x_max=bm.pi,
        offset_len=0.0,
    ):
        super(GD_cell, self).__init__()

        # dynamics parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v  # The time constant of the adaptation variable
        self.ratio = ratio
        self.num_x = num  # number of excitatory neurons for x dimension
        self.num_y = num  # number of excitatory neurons for y dimension
        self.num = self.num_x * self.num_y
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0  # maximum connection value
        self.m = mbar * tau / tau_v
        self.noise_stre = noise_stre
        self.offset_len = offset_len

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
        # self.coor_transform = bm.array([[1 , -1/bm.sqrt(3)],[0, 2/bm.sqrt(3)]])
        self.coor_transform = bm.array([[1, 0], [0, 1]])

        # initialize conn matrix
        self.conn_mat = self.make_conn()

        # initialize dynamical variables
        self.r = bm.Variable(bm.zeros(self.num))
        self.u = bm.Variable(bm.zeros(self.num))
        self.v = bm.Variable(bm.zeros(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.center = bm.Variable(bm.zeros(2))
        self.center_I = bm.Variable(bm.zeros(2))

        # 定义积分器
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    def circle_period(self, d):
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        d = bm.where(d < -bm.pi, d + 2 * bm.pi, d)
        return d

    def dist(self, d):
        d = self.circle_period(d)
        # delta_x = d[:,0] + d[:,1]/2
        # delta_y = d[:,1] * bm.sqrt(3)/2
        delta_x = d[:, 0]
        delta_y = d[:, 1]
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

    def get_input_via_conjG(self, Animal_location, Internal_direction):
        assert bm.size(Animal_location) == 2
        offset = bm.array([bm.cos(Internal_direction), bm.sin(Internal_direction)]) * self.offset_len
        self.center_conjG = self.Postophase(
            Animal_location.reshape(-1,) + offset.reshape(-1,)
        )  # Ideal phase using mapping function
        d = self.dist(bm.asarray(self.center_conjG) - self.value_grid)
        return self.A / 5 * bm.exp(-0.25 * bm.square(d / self.a))

    def get_input_via_posistion(self, Animal_location):
        assert bm.size(Animal_location) == 2
        self.centerI = self.Postophase(
            Animal_location.reshape(-1,)
        )  # Ideal phase using mapping function
        d = self.dist(bm.asarray(self.centerI) - self.value_grid)
        return self.A * bm.exp(-0.25 * bm.square(d / self.a))

    def get_center(self):
        exppos_x = bm.exp(1j * self.x_grid)
        exppos_y = bm.exp(1j * self.y_grid)
        r = bm.where(self.r > bm.max(self.r) * 0.1, self.r, 0)
        self.center[0] = bm.angle(bm.sum(exppos_x * r))
        self.center[1] = bm.angle(bm.sum(exppos_y * r))

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

    def update(self, Animal_location, Internal_direction, ThetaInput):
        self.get_center()
        input_conjG = self.input_by_conjG(pos, hd)
        
        input_pos = self.get_input_via_posistion(Animal_location)
        input_conjG = self.get_input_via_conjG(Animal_location, Internal_direction)
        #no direct input from the environment, but only from the conjunctive grid cells
        self.input = (0*input_pos + input_conjG)*ThetaInput
        
        Irec = bm.matmul(self.conn_mat, self.r) + self.noise_stre * bm.random.randn(
            (self.num)
        )
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share["t"], Irec, bm.dt)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2


def calculate_inst_speed(directions, samples_per_sec):
    diff_dist = np.diff(directions.flatten())
    # consider the periodic boundary condition that is, if diff > pi, then diff = diff - 2*pi
    # if diff < -pi, then diff = diff + 2*pi
    diff_dist = np.where(diff_dist > np.pi, diff_dist - 2 * np.pi, diff_dist)
    diff_dist = np.where(diff_dist < -np.pi, diff_dist + 2 * np.pi, diff_dist)
    inst_speed = diff_dist * samples_per_sec
    # insert the first element the same as the second element
    inst_speed = np.insert(inst_speed, 0, 0)
    return inst_speed


def create_directory_if_not_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def period(data):
    # 计算傅里叶变换
    fft_x = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data), d=(bm.dt))

    # 仅使用正频率部分
    positive_frequencies = frequencies[np.where(frequencies >= 0)]
    positive_fft_x = np.abs(fft_x[np.where(frequencies >= 0)])

    # 找到最大频率分量
    dominant_frequency = positive_frequencies[np.argmax(positive_fft_x)]
    dominant_period = 1 / dominant_frequency

    # 打印结果
    print(f"Dominant Frequency: {dominant_frequency} Hz")
    print(f"Dominant Period: {dominant_period} ")


def circle_period(d):
    d = np.where(d > np.pi, d - 2 * np.pi, d)
    d = np.where(d < -np.pi, d + 2 * np.pi, d)
    return d


def traj(x0, v, T):
    x = []
    xt = x0
    for i in range(T):
        xt = xt + v * bm.dt
        if xt > np.pi:
            xt -= 2 * np.pi
        if xt < -np.pi:
            xt += 2 * np.pi
        x.append(xt)
    return np.array(x)

class Head_direction_cell(bp.DynamicalSystem):
    def __init__(
        self,
        num,
        noise_stre=0.01,
        tau=1.0,
        tau_v=10.0,
        k=0.8,
        mbar=2,
        a=bm.pi / 4,
        A=3.0,
        J0=5.0,
        z_min=-bm.pi,
        z_max=bm.pi,
    ):
        super(Head_direction_cell, self).__init__()

        # parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v
        self.k = k / num * 20  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0 / num * 20  # maximum connection value
        self.m = mbar * tau / tau_v
        self.noise_stre = noise_stre

        # neuron num
        self.num = num  # head-direction cell
        # feature space
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        x1 = bm.linspace(z_min, z_max, num + 1)  # The encoded feature values
        self.x = x1[0:-1]
        self.rho = num / self.z_range  # The neural density
        self.dx = self.z_range / num  # The stimulus density

        # connection matrix
        conn_mat = self.make_conn()
        self.conn_fft = bm.fft.fft(conn_mat)

        # neuron state variables
        self.r = bm.Variable(bm.zeros(num))
        self.u = bm.Variable(bm.zeros(num))  # head direction cell
        self.v = bm.Variable(bm.zeros(num))
        self.input = bm.Variable(bm.zeros(num))
        self.center = bm.Variable(bm.zeros(1))
        self.centerI = bm.Variable(bm.zeros(1))

        # 定义积分器
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    def dist(self, d):
        d = self.circle_period(d)
        d = bm.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self):
        d = self.dist(bm.abs(self.x[0] - self.x))
        Jxx = self.J0 * bm.exp(-0.5 * bm.square(d / self.a)) / (2 * bm.pi * self.a ** 2)
        return Jxx

    def get_center(self, r, x):
        exppos = bm.exp(1j * x)
        center = bm.angle(bm.sum(exppos * r))
        return center.reshape(-1,)

    @property
    def derivative(self):
        du = lambda u, t, input: (-u + input - self.v) / self.tau
        dv = lambda v, t: (-v + self.m * self.u) / self.tau_v
        return bp.JointEq([du, dv])

    def circle_period(self, A):
        B = bm.where(A > bm.pi, A - 2 * bm.pi, A)
        B = bm.where(B < -bm.pi, B + 2 * bm.pi, B)
        return B

    def input_HD(self, HD):
        # integrate self motion
        return self.A * bm.exp(-0.25 * bm.square(self.dist(self.x - HD) / self.a))

    def reset_state(self, HD_truth):
        self.r.value = bm.Variable(bm.zeros(self.num))
        self.u.value = bm.Variable(bm.zeros(self.num))
        self.v.value = bm.Variable(bm.zeros(self.num))
        self.center.value = bm.Variable(bm.zeros(1,) + HD_truth)

    def update(self, HD):
        self.center = self.get_center(r=self.r, x=self.x)
        Iext = self.input_HD(HD)
        # Calculate input
        r_fft = bm.fft.fft(self.r)
        Irec = bm.real(bm.fft.ifft(r_fft * self.conn_fft))
        input_total = Iext + Irec + bm.random.randn(self.num) * self.noise_stre
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share.load("t"), input_total, bm.dt)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2


# Grid cell model modules
class Grid_cells(bp.DynamicalSystem):
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
        A_theta=0.1,
        T_theta=20,
        J0=5.0,
        k=1,
        x_min=-bm.pi,
        x_max=bm.pi,
        offset_len=0.0,
    ):
        super(Grid_cells, self).__init__()

        # dynamics parameters
        self.tau = tau  # The synaptic time constant
        self.tau_v = tau_v  # The time constant of the adaptation variable
        self.ratio = ratio
        self.num_x = num  # number of excitatory neurons for x dimension
        self.num_y = num  # number of excitatory neurons for y dimension
        self.num = self.num_x * self.num_y
        self.k = k  # Degree of the rescaled inhibition
        self.a = a  # Half-width of the range of excitatory connections
        self.A = A  # Magnitude of the external input
        self.J0 = J0  # maximum connection value
        self.m = mbar * tau / tau_v
        self.noise_stre = noise_stre

        #
        self.T = T_theta
        self.A_theta = A_theta
        self.offset_len = offset_len

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
        # self.coor_transform = bm.array([[1 , -1/bm.sqrt(3)],[0, 2/bm.sqrt(3)]])
        self.coor_transform = bm.array([[1, 0], [0, 1]])

        # initialize conn matrix
        self.conn_mat = self.make_conn()

        # initialize dynamical variables
        self.r = bm.Variable(bm.zeros(self.num))
        self.u = bm.Variable(bm.zeros(self.num))
        self.v = bm.Variable(bm.zeros(self.num))
        self.input = bm.Variable(bm.zeros(self.num))
        self.center = bm.Variable(bm.zeros(2))
        self.center_I = bm.Variable(bm.zeros(2))

        # 定义积分器
        self.integral = bp.odeint(method="exp_euler", f=self.derivative)

    def circle_period(self, d):
        d = bm.where(d > bm.pi, d - 2 * bm.pi, d)
        d = bm.where(d < -bm.pi, d + 2 * bm.pi, d)
        return d

    def dist(self, d):
        d = self.circle_period(d)
        # delta_x = d[:,0] + d[:,1]/2
        # delta_y = d[:,1] * bm.sqrt(3)/2
        delta_x = d[:, 0]
        delta_y = d[:, 1]
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

    def input_by_conjG(self, pos, hd):
        assert bm.size(pos) == 2
        offset = bm.array([bm.cos(hd), bm.sin(hd)]) * self.offset_len
        self.center_conjG = self.Postophase(
            pos.reshape(-1,) + offset.reshape(-1,)
        )  # Ideal phase using mapping function
        d = self.dist(bm.asarray(self.center_conjG) - self.value_grid)
        return self.A / 5 * bm.exp(-0.25 * bm.square(d / self.a))

    def get_stimulus_by_pos(self, pos):
        assert bm.size(pos) == 2
        self.centerI = self.Postophase(
            pos.reshape(-1,)
        )  # Ideal phase using mapping function
        d = self.dist(bm.asarray(self.centerI) - self.value_grid)
        return self.A * bm.exp(-0.25 * bm.square(d / self.a))

    def get_center(self):
        exppos_x = bm.exp(1j * self.x_grid)
        exppos_y = bm.exp(1j * self.y_grid)
        r = bm.where(self.r > bm.max(self.r) * 0.1, self.r, 0)
        self.center[0] = bm.angle(bm.sum(exppos_x * r))
        self.center[1] = bm.angle(bm.sum(exppos_y * r))

    @property
    def derivative(self):
        du = (
            lambda u, t, Irec: (
                -u
                + Irec
                + self.input
                - self.v
                + self.A_theta * bm.cos((t * 2 * bm.pi + self.phase) / (self.T))
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

    def update(self, pos, hd=0.0, conjG=0):
        self.get_center()
        input_conjG = self.input_by_conjG(pos, hd)
        self.input = self.get_stimulus_by_pos(pos) + bm.where(
            conjG == 1, input_conjG, 0
        )
        Irec = bm.matmul(self.conn_mat, self.r) + self.noise_stre * bm.random.randn(
            (self.num)
        )
        # Update neural state
        u, v = self.integral(self.u, self.v, bp.share["t"], Irec, bm.dt)
        self.u.value = bm.where(u > 0, u, 0)
        self.v.value = v
        r1 = bm.square(self.u)
        r2 = 1.0 + self.k * bm.sum(r1)
        self.r.value = r1 / r2


def calculate_inst_speed(directions, samples_per_sec):
    diff_dist = np.diff(directions.flatten())
    # consider the periodic boundary condition that is, if diff > pi, then diff = diff - 2*pi
    # if diff < -pi, then diff = diff + 2*pi
    diff_dist = np.where(diff_dist > np.pi, diff_dist - 2 * np.pi, diff_dist)
    diff_dist = np.where(diff_dist < -np.pi, diff_dist + 2 * np.pi, diff_dist)
    inst_speed = diff_dist * samples_per_sec
    # insert the first element the same as the second element
    inst_speed = np.insert(inst_speed, 0, 0)
    return inst_speed


def create_directory_if_not_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def period(data):
    # 计算傅里叶变换
    fft_x = np.fft.fft(data)
    frequencies = np.fft.fftfreq(len(data), d=(bm.dt))

    # 仅使用正频率部分
    positive_frequencies = frequencies[np.where(frequencies >= 0)]
    positive_fft_x = np.abs(fft_x[np.where(frequencies >= 0)])

    # 找到最大频率分量
    dominant_frequency = positive_frequencies[np.argmax(positive_fft_x)]
    dominant_period = 1 / dominant_frequency

    # 打印结果
    print(f"Dominant Frequency: {dominant_frequency} Hz")
    print(f"Dominant Period: {dominant_period} ")


def circle_period(d):
    d = np.where(d > np.pi, d - 2 * np.pi, d)
    d = np.where(d < -np.pi, d + 2 * np.pi, d)
    return d


def traj(x0, v, T):
    x = []
    xt = x0
    for i in range(T):
        xt = xt + v * bm.dt
        if xt > np.pi:
            xt -= 2 * np.pi
        if xt < -np.pi:
            xt += 2 * np.pi
        x.append(xt)
    return np.array(x)
