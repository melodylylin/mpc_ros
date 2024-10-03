#!/usr/bin/env cyecca_python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Joy
import casadi as ca
import numpy as np
from mpc_ros.dubin_planner import gen_waypoints, gen_reference_trajectory
from mpc_ros.NVP_MPC import derive_dynamics, update_param, nlp_multiple_shooting_cbf
from cyecca.lie import SE2, SO3EulerB321, SO3Quat

# Planner
from mpc_ros.plan_dubins import plan_dubins_path
from mpc_ros.plan_rrtstar import RRTStar
import minsnap_trajectories as ms
from mpc_ros.plan_poly import plan_poly_traj

# rrt*
map_bound = [-10,10]

start = [0.0, 0.0, 0.0]
goal = [6.5, -7.5, -np.pi/2]
obs_r = 1
robot_r = 1
plan_time = 1

obs_list = [(4,0, obs_r), 
            (8,5, obs_r),
            (6,9, obs_r), 
            (2, -4, obs_r), 
            (8,-5, obs_r), 
            (6,-9, obs_r), 
            (5, -6, obs_r)
        ]
#path = [[start[0], start[1]]]
# ====Search Path with RRT====
# Set Initial parameters
rrt_star = RRTStar(
    start=[start[0], start[1]],
    goal=[goal[0], goal[1]],
    rand_area=map_bound,
    obstacle_list=obs_list,
    expand_dis=3.0,
    robot_radius=0.5,
    max_iter=1000,
    search_until_max_iter=True)
path = rrt_star.planning(animation=False)
path = np.array(path)
path = np.unique(path, axis=0).tolist()
if path is None:
    print("Cannot find path")
else:
    print("found path!!")     

# dubin
r = 2
v = 1.5
dt = 0.1

traj_x_dubins = np.array([])
traj_y_dubins = np.array([])
traj_theta_dubins = np.array([])

for i in range(len(path)-1):
    # First Leg
    if i == 0:
        last_yaw = start[2]
        next_yaw = np.arctan2(path[i+1][1] - path[i][1], path[i+1][0] - path[i][0])
        
    # Last Leg
    elif i == len(path)-2:
        last_yaw = path_theta[-1]
        next_yaw = goal[2]

    # Intermediate Leg
    else:
        last_yaw = path_theta[-1]
        next_yaw = np.arctan2(path[i+1][1] - path[i][1], path[i+1][0] - path[i][0])
    
    path_x, path_y, path_theta, _, _ = plan_dubins_path(path[i][0], path[i][1], last_yaw, path[i+1][0], path[i+1][1], next_yaw, r,
                    step_size=v*dt)    
    
    traj_x_dubins = np.concatenate((traj_x_dubins,path_x))
    traj_y_dubins = np.concatenate((traj_y_dubins,path_y))
    traj_theta_dubins = np.concatenate((traj_theta_dubins,path_theta))

traj_vx_dubins = [0.0]
traj_vy_dubins = [0.0]
traj_omega_dubins = [0.0]
traj_time = [0.0]
time = 0
for i in range(traj_x_dubins.shape[0]-1):
    vx = (traj_x_dubins[i+1] - traj_x_dubins[i])/dt
    vy= (traj_y_dubins[i+1] - traj_y_dubins[i])/dt
    omega = (traj_theta_dubins[i+1] - traj_theta_dubins[i])/dt
    time+=dt
    traj_vx_dubins += [vx]
    traj_vy_dubins += [vy]
    traj_omega_dubins += [omega]
    traj_time  += [time]

traj_time = np.array(traj_time)    
traj_vx_dubins = np.array(traj_vx_dubins)
traj_vy_dubins = np.array(traj_vy_dubins)
traj_omega_dubins = np.array(traj_omega_dubins)

traj_dubins = {
    't': traj_time,
    'x': traj_x_dubins,
    'y': traj_y_dubins,
    'theta': traj_theta_dubins,
    'vx': traj_vx_dubins,
    'vy': traj_vy_dubins,
    'omega': traj_omega_dubins,
    }

# poly
num_points = 3
boundary_cdn = []
idx = 0
for i in range(num_points):
    wp = ms.Waypoint(
        time=traj_dubins['t'][idx],
        position=np.array([traj_dubins['x'][idx], traj_dubins['y'][idx]]),
        velocity=np.array([traj_dubins['vx'][idx+1], traj_dubins['vy'][idx+1]]),
    )
    boundary_cdn.append(wp)
    idx = int(idx + traj_dubins['t'].shape[0]/num_points)

idx = int(traj_dubins['t'].shape[0]-1)
goal_wp = ms.Waypoint(
        time=traj_dubins['t'][idx],
        position=np.array([traj_dubins['x'][idx], traj_dubins['y'][idx]]),
        velocity=np.array([traj_dubins['vx'][idx-1],traj_dubins['vy'][idx-1]]),
)
boundary_cdn.append(goal_wp)  

position, velocity = plan_poly_traj(boundary_cdn,traj_dubins['t'][idx], dt)
traj_x_poly = position[:, 0]
traj_y_poly = position[:, 1]

traj_theta_poly = np.arctan2(
                velocity[:, 1], velocity[:, 0]
            )

xt = np.array([traj_x_poly.tolist(), traj_y_poly.tolist(), traj_theta_poly.tolist()]).T

eqs = derive_dynamics()

# Cost Weight
Q_x = 1
Q_y = 1
Q_theta = 0.05
R_v = 0.5
R_omega = 0.1
Q = ca.diagcat(Q_x, Q_y, Q_theta)
R = ca.diagcat(R_v, R_omega)

N = 50
dt = 0.2
nlp = nlp_multiple_shooting_cbf(eqs, N, dt, Q, R)
solver = ca.nlpsol('solver', 'ipopt', nlp['nlp_prob'], nlp['opts'])

n_x = nlp['n_x']
n_u = nlp['n_u']
obstacles = nlp['obstacles']

v_max = 1.5
r_c_max = ca.pi/4
v_min = -v_max
r_c_min = -r_c_max
#### condition without cbf ####
# lbg = ca.DM.zeros((n_x * (N+1)))
# ubg = -ca.DM.zeros((n_x * (N+1)))

#### condition with cbf ####
lbg = ca.DM.zeros((n_x * (N + 1) + len(obstacles)*(N), 1))
ubg = ca.DM.zeros((n_x * (N + 1) + len(obstacles)*(N), 1))
lbg[n_x * (N + 1):] = -ca.inf
ubg[n_x * (N + 1):] = 0

lbx = ca.DM.zeros((n_x * (N + 1) + n_u * N, 1))
ubx = ca.DM.zeros((n_x * (N + 1) + n_u * N, 1))

# states x: [w, r, vx, vy, omega, px, py, theta]
lbx[0:n_x * (N + 1):n_x] = -ca.inf
lbx[1:n_x * (N + 1):n_x] = -ca.inf
lbx[2:n_x * (N + 1):n_x] = -ca.inf

ubx[0:n_x * (N + 1):n_x] = ca.inf
ubx[1:n_x * (N + 1):n_x] = ca.inf
ubx[2:n_x * (N + 1):n_x] = ca.inf

lbx[n_x * (N + 1):n_x * (N + 1) + n_u * N:n_u] = v_min
ubx[n_x * (N + 1):n_x * (N + 1) + n_u * N:n_u] = v_max
lbx[n_x * (N + 1) + 1:n_x * (N + 1) + n_u * N:n_u] = r_c_min
ubx[n_x * (N + 1) + 1:n_x * (N + 1) + n_u * N:n_u] = r_c_max

args = {
    'lbg': lbg,
    'ubg': ubg,
    'lbx': lbx,
    'ubx': ubx,
}

u0 = ca.DM.zeros((n_u, N))

x0 = ca.DM.zeros(n_x, 1)

p = ca.vertcat(1)


class NVPPublisher(Node):
    def __init__(self):
        super().__init__('night_vapor_publisher')
        self.pub_control_input = self.create_publisher(Twist, '/cmd_vel', 10)
        # self.pub_joy = self.create_publisher(Joy, '/auto_joy', 10)
        self.sub_mocap = self.create_subscription(Odometry, '/odometry_b3rb', self.pose_cb, 10)
        self.vehicle_pred_path_pub = self.create_publisher(
            Path, '/vehicle_pred_path', 10
        )
        self.pub_ref_point = self.create_publisher(PoseStamped, '/ref_point', 10)
        self.pub_ref_path = self.create_publisher(Path, '/ref_path', 10)
        self.pub_path = self.create_publisher(Path, '/path', 10)
        self.timer_path = self.create_timer(1, self.publish_ref_path)
        self.timer_velocity = self.create_timer(dt, self.publish_cmd_vel)
        self.vehicle_pred_path_msg = Path()
        self.y = 0.0
        self.theta = 0.0
        self.idx = 0
        self.x_list = []
        self.y_list = []
        self.theta_list = []
        self.trail_size = 1000
        self.tracking_error = []

    def pose_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        orientation_q = msg.pose.pose.orientation
        orientation_list = ca.vertcat(orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z)
        yaw = float(SO3EulerB321.from_Quat(SO3Quat.elem(orientation_list)).param[0])
        self.theta = yaw
            
    def publish_cmd_vel(self):

        state_0 = np.array([self.x, self.y, self.theta])
        x_ref = xt[self.idx, 0]
        y_ref = xt[self.idx, 1]
        theta_ref = xt[self.idx, 2]

        x = SE2.elem(ca.vertcat(self.x, self.y, self.theta))
        xr = SE2.elem(ca.vertcat(x_ref, y_ref, theta_ref))
        xi = (xr.inverse()*x).log()

        self.tracking_error.append(np.sqrt((self.x - x_ref)**2 +(self.y - y_ref)**2)) 
        if (np.sqrt((self.x - xt[-1,0])**2 +(self.y - xt[-1, 1])**2)<1e-2):
            avg_track_error = np.mean(np.array(self.tracking_error))
            print('tracking error',avg_track_error) 

        if (float(xi.param[0]) > -1e-1) and (self.idx < (traj_x_poly.shape[0])-1):
            self.idx += 1
        args['P'] = ca.vertcat(p, update_param(state_0, xt, self.idx, N))
        X0 = ca.repmat(state_0, 1, N + 1)

        args['x0'] = ca.vertcat(ca.reshape(X0, n_x * (N + 1), 1),
                                ca.reshape(u0, n_u * N, 1))

        sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                    lbg=args['lbg'], ubg=args['ubg'], p=args['P'])
        u = ca.reshape(sol['x'][n_x * (N + 1):], n_u, N)
        X = ca.reshape(sol['x'][: n_x * (N + 1)], n_x, N+1)

        self.u0 = ca.horzcat(u[:, 1:],ca.reshape(u[:, -1], -1, 1))

        cmd_vel = u[:,0]
        vel_msg = Twist()
        vel_msg.linear.x = float(cmd_vel[0])
        vel_msg.angular.z = float(cmd_vel[1])
        self.pub_control_input.publish(vel_msg)

        msg_path = Path()
        msg_path.header.frame_id = 'map'
        msg_path.header.stamp = self.get_clock().now().to_msg()
        for i in range(N):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(X[0,i])
            pose.pose.position.y = float(X[1,i])
            pose.pose.orientation.w = np.cos(float(X[2,i])/2)
            pose.pose.orientation.z = np.sin(float(X[2,i])/2)
            msg_path.poses.append(pose)
        self.vehicle_pred_path_pub.publish(msg_path)

        # print("state_x: {:5.2f} state_y: {:5.2f} state_theta: {:5.2f}\t idx: {:5.2f}\t cost: {:5.2f}\t cmd_vel_v: {:5.2f} cmd_vel_rud: {:5.2f}\n"
        #       "ref_x:{:5.2f} ref_y: {:5.2f} ref_theta: {:5.2f}".format(
        #     self.x, self.y, self.theta, self.idx, float(sol['f']), float(cmd_vel[0]), float(cmd_vel[1]), x_ref, y_ref, theta_ref))

        # update next init condition
        # X_next = ca.horzcat(X[:, 1:], ca.reshape(X[:, -1], -1, 1))
        # self.vx = float(X_next[3,0])
        # self.vy = float(X_next[4,0])
        # self.omega = float(X_next[5,0])
            #X0 = casadi.reshape(sol['x'][:n_states * (N + 1)], n_states, N + 1)
            # print(self.throttle, delta)

        ref_pos_msg = PoseStamped()
        ref_pos_msg.header.frame_id = 'map'
        ref_pos_msg.pose.position.x = x_ref
        ref_pos_msg.pose.position.y = y_ref
        theta = theta_ref
        ref_pos_msg.pose.orientation.w = np.cos(theta/2)
        ref_pos_msg.pose.orientation.z = np.sin(theta/2)
        self.pub_ref_point.publish(ref_pos_msg)

        self.x_list.append(self.x)
        self.y_list.append(self.y)
        self.theta_list.append(self.theta)
        #Publish the trajectory of the vehicle
        self.publish_path()

    def publish_ref_path(self):
        msg_path = Path()
        msg_path.header.frame_id = 'map'
        msg_path.header.stamp = self.get_clock().now().to_msg()
        for x, y, theta in zip(traj_x_poly, traj_y_poly, traj_theta_poly):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = np.cos(theta/2)
            pose.pose.orientation.z = np.sin(theta/2)
            msg_path.poses.append(pose)
        self.pub_ref_path.publish(msg_path)

    def publish_path(self):
        msg_path = Path()
        msg_path.header.frame_id = 'map'
        msg_path.header.stamp = self.get_clock().now().to_msg()
        if len(self.x_list) > self.trail_size:
            del self.x_list[0]
            del self.y_list[0]
            del self.theta_list[0]
        for x, y, theta in zip(self.x_list, self.y_list, self.theta_list):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = np.cos(theta/2)
            pose.pose.orientation.z = np.sin(theta/2)
            msg_path.poses.append(pose)
        self.pub_path.publish(msg_path) 
    

def main(args=None):
    rclpy.init(args=args)
    PID_publisher = NVPPublisher()
    rclpy.spin(PID_publisher)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    PID_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
