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
r = 1 # turning radius
n = 100 # number of turning points
x0 = 0
y0 = 0
theta0 = 0
xf = 9
yf = -9
control_point = [(x0, y0), (1.5, 0)]#, (4, -4), (8, 0), (xf, yf)]
wp = gen_waypoints(control_point, r, n)
ref_first = gen_reference_trajectory(wp, 1, 0.2) # wp, vel, dt
ref_final = np.repeat(np.array([[5.5, 0.0, 0.0, 0.0]]), 50, axis=0)
ref = np.vstack([ref_first, ref_final])
ref_x_list = ref[:,0].tolist()
ref_y_list = ref[:,1].tolist()
ref_theta_list = ref[:,2].tolist()
xt = np.array([ref[:,0], ref[:,1], ref[:,2]]).T

eqs = derive_dynamics()

# Cost Weight
Q_x = 5
Q_y = 5
Q_theta = 0.5
R_v = 0.5
R_omega = 0.05
Q = ca.diagcat(Q_x, Q_y, Q_theta)
R = ca.diagcat(R_v, R_omega)

N = 20
dt = 0.2
nlp = nlp_multiple_shooting_cbf(eqs, N, dt, Q, R)
solver = ca.nlpsol('solver', 'ipopt', nlp['nlp_prob'], nlp['opts'])

n_x = nlp['n_x']
n_u = nlp['n_u']
obstacles = nlp['obstacles']

v_max = 1
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

        # if (float(xi.param[0]) > -1e-1) and (self.idx < len(ref_x_list)-1):
        if (self.idx == 56):
            self.idx = self.idx
        else:
            self.idx += 1
        print(self.idx)
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
        print(cmd_vel[0])
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
        for x, y, theta in zip(ref_x_list, ref_y_list, ref_theta_list):
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
