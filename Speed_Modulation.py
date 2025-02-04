import os
import shutil
import numpy as np
import brainpy.math as bm
import matplotlib.pyplot as plt
from Network_simulation import runNetwork
# simulation time step
bm.set_platform('gpu')
bm.set_dt(1.)

# bm.random.seed(123)
# np.random.seed(123)

plot = True
if plot == True:
   figurefolder = './test_figures/SpeedModulation/'
   if os.path.exists(figurefolder):
      shutil.rmtree(figurefolder)
   os.makedirs(figurefolder)

num_speed = 10
all_speed_gain = np.linspace(0.3, 1.5, num_speed)

sweep_results = {}
sweep_results['speed'] = {}
sweep_results['peak_dist'] = {}
sweep_results['trough_dist'] = {}
sweep_results['gc_sweep_angle'] = {}
sweep_results['hd_sweep_angle'] = {}

for i, speed_g in enumerate(all_speed_gain):
    print('*'*100)
    print('Animal speed:', speed_g) 
    
    #run simulation
    results= runNetwork(N=4, speed_gain=speed_g, 
                        A_modulation = 1, # default 1; maximal 3
                        theta_hd_modulation = 0.4, # default 0.4; maximal 0.4
                        theta_gc_modulation = 0.5, # default 0.5; maximal 0.5
                        plot='simpleplot',
                        speedmodulation = ['phase','input'])

    #append results
    sweep_results['speed'][i] = speed_g
    sweep_results['peak_dist'][i] = results['peak_sweep_dist']
    sweep_results['trough_dist'][i] = results['trough_sweep_dist']
    sweep_results['gc_sweep_angle'][i] = results['gc_sweep_angle']
    sweep_results['hd_sweep_angle'][i] = results['hd_sweep_angle']
    #save fig
    fig = results['fig']
    
    #save fig
    fig.savefig(figurefolder + 'GC_sweep_angle_speed_' + str(np.round(speed_g,1)) + '.png')
    
    plt.close(fig)
# change sweep_results['speed'] to numpy array
speeds = np.array(list(sweep_results['speed'].values()))

MeanSweepAngle_gc = []
StdSweepAngle_gc = []
MeanSweepDist = []
AlternationScore_gc = []

MeanSweepAngle_hd = []
StdSweepAngle_hd = []
AlternationScore_hd = []

for i in range(len(speeds)):
    #get sweep distance
    peak_dist = sweep_results['peak_dist'][i]
    trough_dist = sweep_results['trough_dist'][i]
    #mean sweep dist
    mean_sweep_dist = np.mean(peak_dist)-np.mean(trough_dist)
    
    #get sweep angle
    gc_sweep_angle = sweep_results['gc_sweep_angle'][i]
    #flip the odd indices of gc_sweep_angle to the opposite sign
    gc_sweep_angle_flip = gc_sweep_angle.copy()
    gc_sweep_angle_flip[1::2] = -gc_sweep_angle_flip[1::2]
    mean_sweep_angle_gc = np.mean(np.abs(gc_sweep_angle_flip))*180/np.pi
    std_sweep_angle_gc = np.std(np.abs(gc_sweep_angle_flip))*180/np.pi
    
    hd_sweep_angle = sweep_results['hd_sweep_angle'][i]
    
    #flip the odd indices of hd_sweep_angle to the opposite sign
    hd_sweep_angle_flip = hd_sweep_angle.copy()
    hd_sweep_angle_flip[1::2] = -hd_sweep_angle_flip[1::2]
    mean_sweep_angle_hd = np.mean(np.abs(hd_sweep_angle_flip)) #already in degrees when saved
    std_sweep_angle_hd = np.std(np.abs(hd_sweep_angle_flip)) #already in degrees when saved
    
    #get alternation score, which is a triplet of gc_sweep_angle [a,b,c], a is the first angle, b is the second angle, c is the third angle
    #alternation score = |(b-a) - (c-b)|/(2*max(|b-a|, |c-b|))
    alternation_score_gc = []
    for i in range(len(gc_sweep_angle)-2):
        a = gc_sweep_angle[i]
        b = gc_sweep_angle[i+1]
        c = gc_sweep_angle[i+2]
        alternation_score_gc.append(np.abs((b-a)-(c-b))/(2*np.max([np.abs(b-a), np.abs(c-b)])))
    mean_alternation_score_gc = np.mean(alternation_score_gc)
    
    alternation_score_hd = []
    for i in range(len(hd_sweep_angle)-2):
        a = hd_sweep_angle[i]
        b = hd_sweep_angle[i+1]
        c = hd_sweep_angle[i+2]
        alternation_score_hd.append(np.abs((b-a)-(c-b))/(2*np.max([np.abs(b-a), np.abs(c-b)]))) 
    
    #save
    MeanSweepAngle_gc.append(mean_sweep_angle_gc)
    StdSweepAngle_gc.append(std_sweep_angle_gc)
    MeanSweepDist.append(mean_sweep_dist)
    AlternationScore_gc.append(mean_alternation_score_gc)
        
    MeanSweepAngle_hd.append(mean_sweep_angle_hd)
    StdSweepAngle_hd.append(std_sweep_angle_hd)
    AlternationScore_hd.append(np.mean(alternation_score_hd))
        
fig, axs = plt.subplots(2, 2, figsize=(5,3.5), dpi=300)
labelsize = 10
ticksize = 8

ax = axs[0,0]
ax.plot(speeds, AlternationScore_gc, color='black', linestyle='-', marker='o', markersize=5, linewidth=1,
        markerfacecolor='#38c7ff', markeredgewidth=0.5, markeredgecolor='black', label='IL sweep')
ax.plot(speeds, AlternationScore_hd, color='black', linestyle='-', marker='o', markersize=5, linewidth=1,
        markerfacecolor='orange', markeredgewidth=0.5, markeredgecolor='black', label = 'ID sweep')
ax.set_ylabel('Alternation Score', fontsize=labelsize)

#ytick with 2 decimal places
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:.2f}"))

ax.legend(fontsize=ticksize, frameon=False)

ax = axs[0,1]
ax.plot(speeds, MeanSweepAngle_gc, color='black', linestyle='-', marker='o', markersize=5, linewidth=1,
        markerfacecolor='#38c7ff', markeredgewidth=0.5, markeredgecolor='black', label='IL sweep')
ax.plot(speeds, MeanSweepAngle_hd, color='black', linestyle='-', marker='o', markersize=5, linewidth=1,
        markerfacecolor='orange', markeredgewidth=0.5, markeredgecolor='black', label = 'ID sweep')
ax.set_ylabel('Sweep angle (°) \n mean', fontsize=labelsize)

ax.legend(fontsize=ticksize, frameon=False)

ax = axs[1,0]
ax.plot(speeds, StdSweepAngle_gc, color='black', linestyle='-', marker='o', markersize=5, linewidth=1,
        markerfacecolor='#38c7ff', markeredgewidth=0.5, markeredgecolor='black', label='IL sweep')
ax.plot(speeds, StdSweepAngle_hd, color='black', linestyle='-', marker='o', markersize=5, linewidth=1,
        markerfacecolor='orange', markeredgewidth=0.5, markeredgecolor='black', label = 'ID sweep')
#add u00B0 to the y-axis
ax.set_ylabel('Sweep angle (°) \n variance', fontsize=labelsize)

ax.legend(fontsize=ticksize, frameon=False)

ax = axs[1,1]
ax.plot(speeds, MeanSweepDist, color='black', linestyle='-', marker='o', markersize=5, linewidth=1,
        markerfacecolor='#38c7ff', markeredgewidth=0.5, markeredgecolor='black')
ax.set_ylabel('Sweep length (m)', fontsize=labelsize)



for ax in axs.flatten():
    #spines off
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.set_xlabel('Speed (m/s)', fontsize=labelsize)

plt.tight_layout()

plt.savefig('./test_figures/speed_modulation_sweep_features.png', dpi=300)



    