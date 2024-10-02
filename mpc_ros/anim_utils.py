import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def simulate(ref_states, cat_states, cat_controls, 
             obs_list, num_frames, step_horizon, N, reference,
              x_lim, y_lim, save=False):
    
    def create_triangle(state=[0,0,0], h=1, w=0.5, update=False):
        x, y, th = state
        triangle = np.array([
            [h, 0   ],
            [0,  w/2],
            [0, -w/2],
            [h, 0   ]
        ]).T
        rotation_matrix = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th),  np.cos(th)]
        ])

        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]
        
    def create_circle(x, y, obs_r):
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + obs_r * np.cos(np.deg2rad(d)) for d in deg]
        yl = [y + obs_r * np.sin(np.deg2rad(d)) for d in deg]
        return xl, yl
    
    def init():
        return path, horizon, current_state, target_state,

    def animate(i):
        # get variables
        x = cat_states[0, 0, i]
        y = cat_states[1, 0, i]
        th = cat_states[2, 0, i]

        # get ref variables
        x_ref = ref_states[:, 0]
        y_ref = ref_states[:, 1]


        # update ref path
        ref_path.set_data(x_ref, y_ref)

        # update path
        if i == 0:
            path.set_data(np.array([]), np.array([]))
        x_new = np.hstack((path.get_xdata(), x))
        y_new = np.hstack((path.get_ydata(), y))
        path.set_data(x_new, y_new)

        # update horizon
        x_new = cat_states[0, :, i]
        y_new = cat_states[1, :, i]
        horizon.set_data(x_new, y_new)

        # update current_state
        current_state.set_xy(create_triangle([x, y, th], update=True))
        

        u = cat_controls[0, 0, i]
        omega = cat_controls[1, 0, i]
        ctrl_eff = np.sqrt(u**2+omega**2)
        ctrl_eff_new = np.hstack((ctrl_effort.get_ydata(), ctrl_eff))
        time = np.hstack((ctrl_effort.get_xdata(), i*step_horizon))
        ctrl_effort.set_data(time, ctrl_eff_new)
        
        # # update target_state
        # xy = target_state.get_xy()
        # target_state.set_xy(xy)

        return path, horizon, current_state, target_state,

    # create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_xlim(left = x_lim[0], right = x_lim[1])
    ax[0].set_ylim(bottom =y_lim[0], top = y_lim[1])
    ax[0].set_title('State Evolution')
    ax[0].grid()

    ax[1].set_ylim(0 , 5)
    ax[1].set_xlim(0, 25)
    ax[1].set_title('Control Effort')
    ax[1].grid()
    plt.tight_layout()
    # create lines:
    #   path
    path, = ax[0].plot([], [], 'r', linewidth=2)

    ref_path, = ax[0].plot([], [], 'b', linewidth=2)
    #   horizon
    horizon, = ax[0].plot([], [], 'x-g', alpha=0.5)   
    #   current_state
    current_triangle = create_triangle(reference[:3])
    current_state = ax[0].fill(current_triangle[:, 0], current_triangle[:, 1], color='r')
    current_state = current_state[0]
    #   target_state
    target_triangle = create_triangle(reference[3:])
    target_state = ax[0].fill(target_triangle[:, 0], target_triangle[:, 1], color='b')
    target_state = target_state[0]


    #Control Effort
    ctrl_effort, = ax[1].plot([], [], 'r', linewidth=2)

    # Obstacles
    for (ox, oy, obsr) in obs_list:
        circle = plt.Circle((ox, oy), obsr, color='r')
        ax[0].add_patch(circle)

    current_triangle = create_triangle(reference[:3])
    sim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=num_frames,
        #interval=step_horizon*100,
        interval=100,
        blit=True,
        repeat=False
    )

    # if save == True:
    #     sim.save('./animation' + str(time()) +'.gif', writer='ffmpeg', fps=30)

    return sim
