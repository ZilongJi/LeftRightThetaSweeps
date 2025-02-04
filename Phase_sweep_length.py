import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt
import jax
from Network_models import HD_cell, GD_cell_hexagonal, circle_period, GD_cell
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from scipy.signal import find_peaks, peak_widths
import jax

bm.clear_buffer_memory()
devices = jax.devices()
selected_device = devices[0]  # Indexing starts at 0, so 1 selects the second GPU

with jax.default_device(selected_device):
    bm.set_dt(1.)
    num_hd = 100
    noise_stre = 0.1
    v0 = 1.0 * bm.pi / 1000  # baseline speed
    offset_len = 1 / 9
    Animal_speed = 1.0 * v0  # vary this to change the speed of the animal
    A = 5.  # 1.6
    mbar_gc = 15.
    mabr_hd = 12.

    alpha_hd = 0.4
    alpha_gc = 0.6

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
        Animal_location = np.array([x, y])
        return Animal_location

    hd_angle = np.pi / 6
    Animal_location = straight_line(0, v=Animal_speed, angle=hd_angle, T=numT)
    Animal_location = Animal_location.transpose()
    Head_direction = hd_angle * bm.ones(numT)  # fixed head direction, mimicking the animal running in a straight line
    Moving_speed = Animal_speed * bm.ones([numT, 1])

    num_spacing = 5
    Spacing = np.linspace(1.5, 4.5, num_spacing) * np.pi
    Ratio = np.pi * 2 / Spacing
    num_simulations = 1  # Number of simulations for each parameter

    # MEC_model_list = bm.NodeList([])
    # for i in range(num_spacing):
    #     MEC_model_list.append(GD_cell_hexagonal(
    #             ratio=Ratio[i],
    #             A=A,  # 3
    #             a=0.8,
    #             k=1.0,
    #             tau=10.,
    #             tau_v=100.,
    #             mbar=mbar_gc,
    #             noise_stre=noise_stre,
    #             num_hd=num_hd,
    #         ))

    HD_net = HD_cell(num=num_hd,
                    noise_stre=noise_stre / 10,  # gc has much more neurons than hd, 10000 vs 100
                    tau=10., tau_v=100.,
                    k=1., mbar=mabr_hd, a=0.4, A=3., J0=4.,
                    z_min=-bm.pi, z_max=bm.pi)
    

    sweep_length_phase_means = np.zeros(num_spacing)
    sweep_length_phase_stds = np.zeros(num_spacing)
    sweep_length_pos_means = np.zeros(num_spacing)
    sweep_length_pos_stds = np.zeros(num_spacing)

    for i in range(num_spacing):
        print('Spacing:', Spacing[i])
        sweep_lengths_phase = np.zeros(num_simulations)
        sweep_lengths_pos = np.zeros(num_simulations)
        for j in range(num_simulations):
            Grid_net = GD_cell_hexagonal(
                ratio=Ratio[i],
                A=A,  # 3
                a=0.8,
                k=1.0,
                tau=10.,
                tau_v=100.,
                mbar=mbar_gc,
                noise_stre=noise_stre,
                num_hd=num_hd,
            )
            HD_net.reset_state()

            # run coupled net
            def run_CoupleNet(i, Animal_location, Head_direction, Moving_speed):  # 20 x size
                theta_modulation_stre_hd = alpha_hd * 1 / v0 * Moving_speed
                theta_modulation_stre_gc = alpha_gc * 1 / v0 * Moving_speed
                T_theta = 100
                # calculate TheataModulator at time step i
                t = i * bm.dt
                theta_phase = bm.mod(t, T_theta) / T_theta  # theta phase（0,1）
                ThetaModulator_HD = 1 + theta_modulation_stre_hd * bm.cos(theta_phase * 2 * bm.pi)
                ThetaModulator_GC = 1 + theta_modulation_stre_gc * bm.cos(theta_phase * 2 * bm.pi)

                # calculate internal head direction
                HD_net.step_run(i, Head_direction, ThetaModulator_HD)
                HD_activity = HD_net.r
                Phase_Offset = offset_len
                Grid_net.step_run(i, Animal_location, HD_activity, ThetaModulator_GC, Phase_Offset, Moving_speed)
                Animal_phase = Grid_net.Postophase(Animal_location)

                # get results
                GC_bumpcenter = Grid_net.center_pos
                Phase_bumpcenter = Grid_net.center_phase
                return GC_bumpcenter, Phase_bumpcenter, Animal_phase

            center_grid, center_phase, Animal_phase = bm.for_loop(
                run_CoupleNet, (time_steps, Animal_location, Head_direction, Moving_speed), progress_bar=False
            )

            def circle_period(d):
                d = np.where(d > np.pi, d - 2 * np.pi, d)
                d = np.where(d < -np.pi, d + 2 * np.pi, d)
                return d

            start = int(numT / 2)

            dis_x_grid = circle_period(center_phase[start:, 0].reshape(-1) - Animal_phase[start:, 0].reshape(-1))
            dis_y_grid = circle_period(center_phase[start:, 1].reshape(-1) - Animal_phase[start:, 1].reshape(-1))
            dis_vec = bm.array([dis_x_grid, dis_y_grid])
            abs_dis = np.sqrt(dis_x_grid ** 2 + dis_y_grid ** 2)

            sweep_lengths_phase[j] = np.max(abs_dis) - np.min(abs_dis)



            dis_x_grid = center_grid[start:, 0].reshape(-1) - Animal_location[start:, 0].reshape(-1)
            dis_y_grid = center_grid[start:, 1].reshape(-1) - Animal_location[start:, 1].reshape(-1)
            dis_vec = bm.array([dis_x_grid, dis_y_grid])
            abs_dis = np.sqrt(dis_x_grid ** 2 + dis_y_grid ** 2)
            sweep_lengths_pos[j] = np.max(abs_dis) - np.min(abs_dis)

        sweep_length_phase_means[i] = np.mean(sweep_lengths_phase)
        sweep_length_phase_stds[i] = np.std(sweep_lengths_phase)
        sweep_length_pos_means[i] = np.mean(sweep_lengths_pos)
        sweep_length_pos_stds[i] = np.std(sweep_lengths_pos)
        print('Iterative:', i)
        print('sweep_length_means of phase:', sweep_length_phase_means[i])
        print('sweep_length_means of pos:', sweep_length_pos_means[i])


    np.save('sweep_length_means_phase.npy', sweep_length_phase_means)
    np.save('sweep_length_stds_phase.npy', sweep_length_phase_stds)
    np.save('sweep_length_means_pos.npy', sweep_length_pos_means)
    np.save('sweep_length_stds_pos.npy', sweep_length_pos_stds)

    #%%
    plt.figure()
    plt.plot(Spacing, sweep_length_phase_means,label='sweep length in phase space', color='b')
    plt.plot(Spacing, sweep_length_pos_means,label='sweep length in physical space', color='r')
    plt.errorbar(Spacing, sweep_length_phase_means, yerr=sweep_length_phase_stds, fmt='k.', ecolor='b', elinewidth=1, capsize=2)
    plt.errorbar(Spacing, sweep_length_pos_means, yerr=sweep_length_pos_stds, fmt='k.', ecolor='r', elinewidth=1, capsize=2)
    plt.legend()
    # plt.ylim([0.4, 0.6])
    plt.xlabel('Grid Spacing')
    plt.ylabel('Sweep length')
    filename = 'Spacing_dependent_sweep_length_errorbar.png'
    plt.savefig(filename)
    # plt.show()

# %%
