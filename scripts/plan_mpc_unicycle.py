import numpy as np
import casadi
import matplotlib.pyplot as plt
from matplotlib import animation
from plan_dubins import plan_dubins_path

def dm_to_array(dm):
        return np.array(dm.full())

def simulate(ref_states, cat_states, cat_controls, num_frames, step_horizon, N, reference, save=False):
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

    def init():
        return path, horizon#, current_state, target_state,

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

        # # update target_state
        # xy = target_state.get_xy()
        # target_state.set_xy(xy)

        return path, horizon#, current_state, target_state,

    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    min_scale = min(reference[0], reference[1], reference[3], reference[4]) - 2
    max_scale = max(reference[0], reference[1], reference[3], reference[4]) + 2
    ax.set_xlim(left = min_scale, right = max_scale)
    ax.set_ylim(bottom = min_scale, top = max_scale)

    # circle = plt.Circle((obs_x, obs_y), obs_diam/2, color='r')
    # ax.add_patch(circle)

    # create lines:
    #   path
    path, = ax.plot([], [], 'r', linewidth=2)

    ref_path, = ax.plot([], [], 'b', linewidth=2)
    #   horizon
    horizon, = ax.plot([], [], 'x-g', alpha=0.5)
    #   current_state
    current_triangle = create_triangle(reference[:3])
    current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color='r')
    current_state = current_state[0]
    # #   target_state
    # target_triangle = create_triangle(reference[3:])
    # target_state = ax.fill(target_triangle[:, 0], target_triangle[:, 1], color='b')
    # target_state = target_state[0]

    sim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=num_frames,
        #interval=step_horizon*100,
        interval=100,
        blit=False,
        repeat=False
    )

    # if save == True:
    #     sim.save('./animation' + str(time()) +'.gif', writer='ffmpeg', fps=30)
    plt.show()
    return sim

class MPC_CBF_Unicycle:
    def __init__(self, dt ,N, v_lim, omega_lim,  
                 Q, R, cbf_const, 
                 obstacles= None,  obs_diam = 0.5, robot_diam = 0.5, alpha=0.2):
        
        self.dt = dt # Period
        self.N = N  # Horizon Length 
        
        self.Q_x = Q[0]
        self.Q_y = Q[1]
        self.Q_theta = Q[2]
        self.R_v = R[0]
        self.R_omega = R[1]
        self.n_states = 0
        self.n_controls = 0

        self.v_lim = v_lim
        self.omega_lim = omega_lim

        # Initialized in mpc_setup
        self.solver = None
        self.f = None

        self.robot_diam = robot_diam
        self.obs_diam = obs_diam
        self.cbf_const = cbf_const # Bool flag to enable obstacle avoidance
        self.alpha= alpha # Parameter for scalar class-K function, must be positive
        self.obstacles = obstacles

        # Setup with initialization params
        self.setup()

    ## Utilies used in MPC optimization
    # CBF Implementation
    def h_obs(self, state, obstacle, r):
            ox, oy = obstacle
            return ((ox - state[0])**2 + (oy - state[1])**2 - r**2)


    def shift_timestep(self, h, time, state, control):
        delta_state = self.f(state, control[:, 0])
        next_state = casadi.DM.full(state + h * delta_state)
        next_time = time + h
        next_control = casadi.horzcat(control[:, 1:],
                                    casadi.reshape(control[:, -1], -1, 1))

        return next_time, next_state, next_control

    def update_param(self, x0, ref, k, N):
        p = casadi.vertcat(x0)
        for l in range(N):
            if k+l < ref.shape[0]:
                ref_state = ref[k+l, :]
            else:
                ref_state = ref[-1, :]
            xt = casadi.DM([ref_state[0], ref_state[1], ref_state[2]])
            p = casadi.vertcat(p, xt)
        return p
    
    def setup(self):
        x = casadi.SX.sym('x')
        y = casadi.SX.sym('y')
        theta = casadi.SX.sym('theta')
        states = casadi.vertcat(x, y, theta)
        self.n_states = states.numel()

        v = casadi.SX.sym('v')
        omega = casadi.SX.sym('omega')
        controls = casadi.vertcat(v, omega)
        self.n_controls = controls.numel()

        X = casadi.SX.sym('X', self.n_states, self.N + 1)
        U = casadi.SX.sym('U', self.n_controls, self.N)
        P = casadi.SX.sym('P', (self.N + 1) * self.n_states)
        Q = casadi.diagcat(self.Q_x, self.Q_y, self.Q_theta)
        R = casadi.diagcat(self.R_v, self.R_omega)

        rhs = casadi.vertcat(v * casadi.cos(theta), v * casadi.sin(theta), omega)
        self.f = casadi.Function('f', [states, controls], [rhs])

        cost = 0
        g = X[:, 0] - P[:self.n_states]

        for k in range(self.N):
            state = X[:, k]
            control = U[:, k]
            cost = cost + (state - P[(k+1)*self.n_states:(k+2)*self.n_states]).T @ Q @ (state - P[(k+1)*self.n_states:(k+2)*self.n_states]) + \
                    control.T @ R @ control
            next_state = X[:, k + 1]
            k_1 = self.f(state, control)
            k_2 = self.f(state + self.dt/2 * k_1, control)
            k_3 = self.f(state + self.dt/2 * k_2, control)
            k_4 = self.f(state + self.dt * k_3, control)
            predicted_state = state + self.dt/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            g = casadi.vertcat(g, next_state - predicted_state)
        
        if self.cbf_const:
            for k in range(self.N):
                state = X[:, k]
                next_state = X[:, k+1]
                for obs in self.obstacles:    
                    h = self.h_obs(state, obs, (self.robot_diam / 2 + self.obs_diam / 2))
                    h_next = self.h_obs(next_state, obs, (self.robot_diam / 2 + self.obs_diam / 2))
                    g = casadi.vertcat(g,-(h_next-h + self.alpha*h))

        opt_variables = casadi.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

        nlp_prob = {
            'f': cost,
            'x': opt_variables,
            'g': g,
            'p': P
        }

        opts = {
            'ipopt': {
                'sb': 'yes',
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0,
        }
        self.solver = casadi.nlpsol('solver', 'ipopt', nlp_prob, opts)


    def solve(self, X0, u0,  ref, idx):   

        lbx = casadi.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))
        ubx = casadi.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))

        lbx[0:self.n_states * (self.N + 1):self.n_states] = -casadi.inf
        lbx[1:self.n_states * (self.N + 1):self.n_states] = -casadi.inf
        lbx[2:self.n_states * (self.N + 1):self.n_states] = -casadi.inf

        ubx[0:self.n_states * (self.N + 1):self.n_states] = casadi.inf
        ubx[1:self.n_states * (self.N + 1):self.n_states] = casadi.inf
        ubx[2:self.n_states * (self.N + 1):self.n_states] = casadi.inf

        lbx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.v_lim[0]
        ubx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.v_lim[1]
        lbx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.n_controls *self.N:self.n_controls] = self.omega_lim[0]
        ubx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.omega_lim[1]
        
        if self.cbf_const:
            lbg = casadi.DM.zeros((self.n_states * (self.N + 1) + len(self.obstacles)*(self.N), 1))
            ubg = casadi.DM.zeros((self.n_states * (self.N + 1) + len(self.obstacles)*(self.N), 1))

            lbg[self.n_states * (self.N + 1):] = -casadi.inf
            ubg[self.n_states * (self.N + 1):] = 0
        else:
            lbg = casadi.DM.zeros((self.n_states * (self.N+1)))
            ubg = -casadi.DM.zeros((self.n_states * (self.N+1)))

        args = {
            'lbg': lbg,
            'ubg': ubg,
            'lbx': lbx,
            'ubx': ubx
        }
        args['p'] = self.update_param(X0[:,0], ref, idx, self.N)
        args['x0'] = casadi.vertcat(casadi.reshape(X0, self.n_states * (self.N + 1), 1),
                                        casadi.reshape(u0, self.n_controls * self.N, 1))

        sol = self.solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                        lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        u = casadi.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
        X = casadi.reshape(sol['x'][:self.n_states * (self.N + 1)], self.n_states, self.N + 1)
        return u, X 


def main(args=None):
    Q_x = 10
    Q_y = 10
    Q_theta = 10
    R_v = 0.5
    R_omega = 0.005

    dt = 0.1
    N = 20
    idx = 0
    t0 = 0

    x_0 = 0
    y_0 = 0
    theta_0 = 0

    x_goal = 6
    y_goal = 3
    theta_goal = np.pi/2

    r = 1 
    v = 1
    path_x, path_y, path_yaw, _, _ = plan_dubins_path(x_0, y_0, theta_0, x_goal, y_goal, theta_goal, r, step_size=v*dt)

    ref_states = np.array([path_x, path_y, path_yaw]).T

    v_lim = [-1, 1]
    omega_lim = [-casadi.pi/4, casadi.pi/4]
    Q = [Q_x, Q_y, Q_theta]
    R = [R_v, R_omega]
    obs_list = [(4,0), (8,5), (6,9), (2, -4), (8,-5), (6,-9), (5, -6)]

    mpc_cbf = MPC_CBF_Unicycle(dt,N, v_lim, omega_lim, Q, R, obstacles= obs_list, cbf_const=True)
    state_0 = casadi.DM([x_0, y_0, theta_0])
    u0 = casadi.DM.zeros((mpc_cbf.n_controls, N))
    X0 = casadi.repmat(state_0, 1, N + 1)
    cat_states = dm_to_array(X0)
    cat_controls = dm_to_array(u0[:, 0])

    x_arr = [x_0]
    y_arr = [y_0]
    for i in range(len(ref_states)):    
        u, X_pred = mpc_cbf.solve(X0, u0, ref_states, i)

        cat_states = np.dstack((cat_states, dm_to_array(X_pred)))
        cat_controls = np.dstack((cat_controls, dm_to_array(u[:, 0])))
        
        t0, X0, u0 = mpc_cbf.shift_timestep(dt, t0, X_pred, u)
        
        x_arr.append(X0[0,1])
        y_arr.append(X0[1,1])
        idx += 1
    
    num_frames = len(ref_states)
    simulate(ref_states, cat_states, cat_controls, num_frames, dt, N,
         np.array([x_0, y_0, theta_0, x_goal, y_goal, theta_goal]), save=False)


if __name__ == '__main__':
    main()