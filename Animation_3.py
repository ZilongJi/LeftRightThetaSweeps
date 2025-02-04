from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
import numpy as np
import brainpy.math as bm
import matplotlib.pyplot as plt
Env = Environment(params={
    'scale':6,
    "dimensionality": "2D",
})

x1 = 0
x2 = 6
x_bound = np.array([x1, x2, x2, x1, x1])
y_bound = np.array([x1, x1, x2, x2, x1])


v0 = bm.pi/1000 #baseline speed
offset_len = 1/7  * (1/v0) #v0/2
Animal_speed = 1.*v0 #vary this to change the speed of the animal

def generate_trajectory(Animal_speed, dur, ifplot):
    dt = 0.001 #s
    speed_mean = Animal_speed*1e3 #m/s
    rotation_velocity_std = 0 * np.pi/180 #radians
    speed_std = 0 #in m/ms

    Ag = Agent(Env, params = {
        "dt": dt,
        "speed_mean":speed_mean,
        "speed_std": speed_std,
        "rotational_velocity_std": rotation_velocity_std, #set to 120 or 360 will change the tutoriocity of the trajectory
        })

    while Ag.t < dur:
        Ag.update()
        
    Position = Ag.history['pos']
    HeadDirection = Ag.history['head_direction']
    Velocity = Ag.history['vel']

    Position = np.array(Position)
    Velocity = np.array(Velocity)
    Moving_speed = np.linalg.norm(Velocity, axis=1)
    HeadDirection = np.array(HeadDirection)
    # 将 HeadDirection 转换为复数
    complex_numbers = HeadDirection[:, 0] + 1j * HeadDirection[:, 1]
    # 计算复数的幅角
    HD_angle = np.angle(complex_numbers)

    if ifplot == True:
        #plot Position
        fig, ax = plt.subplots(1,3,figsize=(9,3))
        ax[0].plot(x_bound, y_bound, 'k')
        ax[0].plot(Position[:,0], Position[:,1])
        ax[0].axis('equal')   
        ax[0].set_title('Animal location')
        ax[1].plot(Moving_speed)
        ax[1].set_title('Moving speed')
        ax[2].plot(HD_angle)
        ax[2].set_title('Head-direction angle')
        plt.tight_layout()

    return Position, Moving_speed, HD_angle
Position, Moving_speed, HD_angle = generate_trajectory(Animal_speed, dur=10, ifplot=True)
plt.show()