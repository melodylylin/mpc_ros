import casadi as ca
import numpy as np
import time

def shift_timestep(h, state, control, f, p):
    delta_state = f(state, control[:, 0], p)
    next_state = ca.DM.full(state + h * delta_state)
    next_control = ca.horzcat(control[:, 1:],
                                  ca.reshape(control[:, -1], -1, 1))

    return next_state, next_control

def dm_to_array(dm):
    return np.array(dm.full())

def derive_dynamics():
    # p, parameters
    # parameters p: [tau_w, tau_r, c_t, m, J_z, rho, S]
    L = ca.SX.sym('L') # motor time constant
    p = ca.vertcat(L) # parameter vector

    # x, state
    # states x: [w, r, vx, vy, omega, px, py, theta]
    p_x = ca.SX.sym('p_x') # x pos
    p_y = ca.SX.sym('p_y') # y pos
    theta = ca.SX.sym('theta') # vehicle heading angle about z
    x = ca.vertcat(p_x, p_y, theta) #state

    # u, input
    # input u: [w_c, r_c]
    v = ca.SX.sym('v') # forward vel
    r_c = ca.SX.sym('r_c') # rudder deflection command
    u = ca.vertcat(v, r_c)

    # f = rhs d/dt x = f(x, u)
    x_dot = ca.vertcat(
        v*ca.cos(theta),
        v*ca.sin(theta),
        v/L*ca.tan(r_c)
    )
    
    f = ca.Function("f", [x, u, p], [x_dot], ["x", "u", "p"], ["x_dot"])
    return locals()

## single shooting method
# x0 is known
# know x1_p = f(x0, u0)
#
# find optimal u0 such that
# x1 = xt  (for simple case xt = x0)

## multiple shooting method
# x0 is known
# know x1_p = f(x0, u0)
#
# find optimal u0 such that
# x1 = xt  (for simple case xt = x0)

def nlp_multiple_shooting(eqs, N, dt, Q, R):
    f = eqs['f'] # getting dynamics function
    
    n_x = eqs['x'].numel()  # numbef of states
    n_u = eqs['u'].numel()  # number of inputs
    n_p = eqs['p'].numel() # number of param in dynamics
    P = ca.SX.sym('P', n_p+(N+1)*n_x,1)
    p = P[:n_p] # L, constant for dynamics model
    x0 = P[n_p:n_p+n_x]

    x_opt = ca.SX.sym('x_opt', n_x, N+1)
    u_opt = ca.SX.sym('u_opt', n_u, N)
    
    # design vector for optimization
    xd_opt = ca.vertcat(x_opt.reshape((-1, 1)), u_opt.reshape((-1, 1)))
    
    f_cost = 0
    f_constraint = x_opt[:,0] - x0
    
    for k in range(N):
        u = u_opt[:,k]
        x = x_opt[:,k]
        xt = P[n_p+(k+1)*n_x:n_p+(k+2)*n_x] 
        x_next = x_opt[:,k+1]
        # one step of rk4
        
        k_1 = f(x, u, p)
        k_2 = f(x + dt/2 * k_1, u, p)
        k_3 = f(x + dt/2 * k_2, u, p)
        k_4 = f(x + dt * k_3, u, p)
        x1 = x + dt/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        
        # cost is how far we are at the end of the simulation from the desired target
        f_cost = f_cost + (x - xt).T @Q@ (x - xt) + u.T@R@u
        # print('f_cost', f_cost)
        f_constraint = ca.vertcat(f_constraint, x_next - x1) # how far dynamic simulation is off from rk4
    
    nlp_prob = {
        'f': f_cost,
        'x': xd_opt,
        'g': f_constraint,
        'p': P,
    }
    opts = {
        'ipopt':  {
            'max_iter': 2000,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6,
        },
        'print_time': 0,
    }
    return locals()

def h_obs(state, obstacle, r):
        ox, oy = obstacle
        return ((ox - state[0])**2 + (oy - state[1])**2 - r**2)

def nlp_multiple_shooting_cbf(eqs, N, dt, Q, R):
    f = eqs['f'] # getting dynamics function
    
    n_x = eqs['x'].numel()  # numbef of states
    n_u = eqs['u'].numel()  # number of inputs
    n_p = eqs['p'].numel() # number of param in dynamics
    P = ca.SX.sym('P', n_p+(N+1)*n_x,1)
    p = P[:n_p] # L, constant for dynamics model
    x0 = P[n_p:n_p+n_x]

    x_opt = ca.SX.sym('x_opt', n_x, N+1)
    u_opt = ca.SX.sym('u_opt', n_u, N)
    
    # design vector for optimization
    xd_opt = ca.vertcat(x_opt.reshape((-1, 1)), u_opt.reshape((-1, 1)))
    
    f_cost = 0
    f_constraint = x_opt[:,0] - x0

    obstacles = [(4,0)]#, (8,5), (6,9), (2,-4), (8,-5), (6,-9), (5,-6)]
    obs_diam = 2
    alpha= 0.001  #Parameter for scalar class-K function, must be positive
    rob_diam = 0.5
    
    for k in range(N):
        u = u_opt[:,k]
        x = x_opt[:,k]
        xt = P[n_p+(k+1)*n_x:n_p+(k+2)*n_x] 
        x_next = x_opt[:,k+1]
        # one step of rk4
        
        k_1 = f(x, u, p)
        k_2 = f(x + dt/2 * k_1, u, p)
        k_3 = f(x + dt/2 * k_2, u, p)
        k_4 = f(x + dt * k_3, u, p)
        x1 = x + dt/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        
        # cost is how far we are at the end of the simulation from the desired target
        f_cost = f_cost + (x - xt).T @Q@ (x - xt) + u.T@R@u
        # print('f_cost', f_cost)
        f_constraint = ca.vertcat(f_constraint, x_next - x1) # how far dynamic simulation is off from rk4

    for k in range(N):
        state = x_opt[:, k]
        next_state = x_opt[:, k+1]
        for obs in obstacles:    
            h = h_obs(state, obs, (rob_diam / 2 + obs_diam / 2))
            h_next = h_obs(next_state, obs, (rob_diam / 2 + obs_diam / 2))
            f_constraint = ca.vertcat(f_constraint,-(h_next-h + alpha*h))
    
    nlp_prob = {
        'f': f_cost,
        'x': xd_opt,
        'g': f_constraint,
        'p': P,
    }
    opts = {
        'ipopt':  {
            'max_iter': 2000,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6,
        },
        'print_time': 0,
    }
    return locals()

def update_param(x0, ref, k, N):
    p = ca.vertcat(x0)
    for l in range(N):
        if k+l < ref.shape[0]:
            ref_state = ref[k+l, :]
            # v = 1
        else:
            ref_state = ref[-1, :]
            # v = 0
        xt = ca.DM([ref_state[0], ref_state[1], ref_state[2]])
        p = ca.vertcat(p, xt)
    return p

def derive_uni_dynamics():
    p = ca.SX.sym('p')

    # x, state
    # states x: [w, r, vx, vy, omega, px, py, theta]
    p_x = ca.SX.sym('p_x') # x pos
    p_y = ca.SX.sym('p_y') # y pos
    theta = ca.SX.sym('theta') # vehicle heading angle about z
    x = ca.vertcat(p_x, p_y, theta) #state

    # u, input
    # input u: [w_c, r_c]
    v = ca.SX.sym('v') # forward vel
    omega = ca.SX.sym('r_c') # rudder deflection command
    u = ca.vertcat(v, omega)

    # f = rhs d/dt x = f(x, u)
    x_dot = ca.vertcat(
        v*ca.cos(theta),
        v*ca.sin(theta),
        omega
    )
    
    f = ca.Function("f", [x, u, p], [x_dot], ["x", "u", "p"], ["x_dot"])
    return locals()