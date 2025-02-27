#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Joy
import casadi as ca
from tf_transformations import euler_from_quaternion
import numpy as np
from mpc_ros.dubin_planner import gen_waypoints, gen_reference_trajectory
from mpc_ros.NVP_MPC import derive_dynamics, update_param, nlp_multiple_shooting
# from cyecca.lie import SE2, SO3EulerB321, SO3Quat

class SE2:
    
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
    
    def to_matrix(self):
        x = self.x
        y = self.y
        theta = self.theta
        cos = np.cos
        sin = np.sin
        return np.array([
            [cos(theta), -sin(theta), x],
            [sin(theta), cos(theta), y],
            [0, 0, 1]])

    @classmethod
    def from_matrix(cls, m):
        theta = np.arctan2(m[1, 0], m[0, 0])
        x = m[0, 2]
        y = m[1, 2]
        return cls(x=x, y=y, theta=theta)

    def __matmul__(self, other):
        return SE2.from_matrix(self.to_matrix()@other.to_matrix())

    def __repr__(self):
        return 'x {:g}: y: {:g} theta: {:g}'.format(self.x, self.y, self.theta)
    
    def log(self):
        x = self.x
        y = self.y
        theta = self.theta
        if (np.abs(theta) > 1e-2):
            a = np.sin(theta)/theta
            b = (1 - np.cos(theta))/theta
        else:
            a = 1 - theta**2/6 + theta**4/120
            b = theta/2 - theta**3/24 + theta**5/720
        V_inv = np.array([
            [a, b],
            [-b, a]])/(a**2 + b**2)
        u = V_inv@np.array([x, y])
        return SE2(x=u[0], y=u[1], theta=theta)
    
    def inv(self):
        x = self.x
        y = self.y
        theta = self.theta
        t = -np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]])@np.array([x, y])
        return SE2(x=t[0], y=t[1], theta=-theta)


class se2:
    
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
    
    def to_matrix(self):
        x = self.x
        y = self.y
        theta = self.theta
        return np.array([
            [0, -theta, x],
            [theta, 0, y],
            [0, 0, 0]])

    @classmethod
    def from_matrix(cls, m):
        x = m[0, 2]
        y = m[1, 2]
        theta = m[1, 0]
        return cls(x=x, y=y, theta=theta)

    def __repr__(self):
        return 'x {:g}: y: {:g} theta: {:g}'.format(self.x, self.y, self.theta)
    
    def exp(self):
        x = self.x
        y = self.y
        theta = self.theta
        if (np.abs(theta) > 1e-2):
            a = np.sin(theta)/theta
            b = (1 - np.cos(theta))/theta
        else:
            a = 1 - theta**2/6 + theta**4/120
            b = theta/2 - theta**3/24 + theta**5/720
        V = np.array([
            [a, -b],
            [b, a]])
        u = V@np.array([x, y])
        return SE2(x=u[0], y=u[1], theta=theta)

# Planner
r = 1.22 # turning radius
n = 10 # number of turning points

x0 = 0.0
y0 = 1.58
# x0 = 3.74
# y0 = 4.54
theta0 = -2.74#-1.2

xf = -1.69
yf = 5.70

control_point = [(x0, y0), (0.5, 3.0), (-0.29, 4.38), (xf, yf)]
wp = gen_waypoints(control_point, r, n)
ref = gen_reference_trajectory(wp, 2.5, 0.01) # wp, vel, dt

ref_x_list = ref[:,0].tolist()
ref_y_list = ref[:,1].tolist()
ref_v_list = [0.0]
for i in range(ref[:,0].shape[0]-1):
    vx = (ref[i+1,0] - ref[i,0])/0.01
    vy= (ref[i+1,1] - ref[i,1])/0.01
    v = np.sqrt(vx**2 + vy**2)
    ref_v_list.append(v)
ref_theta_list = ref[:,2].tolist()
# for i in range(ref.shape[0]):
#     ref[i, 2] = (ref[i, 2] + np.pi) % (2 * np.pi) - np.pi
xt = np.array([ref_x_list, ref_y_list, ref_theta_list, ref_v_list]).T
print(xt.shape)

eqs = derive_dynamics()

# Cost Weight
Q_x = 150
Q_y = 150
Q_theta = 25
Q_v = 5
R_a = 0.2
R_omega = 0.2
Q = ca.diagcat(Q_x, Q_y, Q_theta, Q_v)
R = ca.diagcat(R_a, R_omega)

N = 2
dt = 0.01
nlp = nlp_multiple_shooting(eqs, N, dt, Q, R)
solver = ca.nlpsol('solver', 'ipopt', nlp['nlp_prob'], nlp['opts'])

n_x = nlp['n_x']
n_u = nlp['n_u']

a_max = 0.35
r_c_max = ca.pi/6
a_min = -0.25
r_c_min = -r_c_max
lbg = ca.DM.zeros((n_x * (N+1)))
ubg = -ca.DM.zeros((n_x * (N+1)))

lbx = ca.DM.zeros((n_x * (N + 1) + n_u * N, 1))
ubx = ca.DM.zeros((n_x * (N + 1) + n_u * N, 1))

# states x: [w, r, vx, vy, omega, px, py, theta]
lbx[0:n_x * (N + 1):n_x] = -ca.inf
lbx[1:n_x * (N + 1):n_x] = -ca.inf
lbx[2:n_x * (N + 1):n_x] = -ca.inf
lbx[3:n_x * (N + 1):n_x] = 0

ubx[0:n_x * (N + 1):n_x] = ca.inf
ubx[1:n_x * (N + 1):n_x] = ca.inf
ubx[2:n_x * (N + 1):n_x] = ca.inf
ubx[3:n_x * (N + 1):n_x] = ca.inf

lbx[n_x * (N + 1):n_x * (N + 1) + n_u * N:n_u] = a_min
ubx[n_x * (N + 1):n_x * (N + 1) + n_u * N:n_u] = a_max
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
        # self.pub_control_input = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_joy = self.create_publisher(Joy, '/nv2/auto_joy', 10)
        # self.sub_mocap = self.create_subscription(Odometry, '/nightvapor2/odom', self.pose_cb, 10)
        self.sub_mocap = self.create_subscription(Odometry, '/nv2/odom', self.pose_cb, 10)
        self.pub_ref_point = self.create_publisher(PoseStamped, '/ref_point2', 10)
        self.pub_ref_path = self.create_publisher(Path, '/ref_path2', 10)
        self.pub_path = self.create_publisher(Path, '/path2', 10)
        self.vehicle_pred_path_pub = self.create_publisher(
            Path, '/vehicle_pred_path', 10
        )
        self.vehicle_pred_path_msg = Path()
        self.timer_path = self.create_timer(1, self.publish_ref_path)
        self.timer = self.create_timer(0.01, self.pub_night_vapor)
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.theta = 0.0
        self.time = 0
        self.dt = 0.01
        self.takeoff = False
        self.taxi = True
        self.takeoff_time = 0.0
        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.theta_list = []
        self.throttle = 1.0
        self.trail_size = 1000
        self.prev_x = None
        self.prev_y = None
        self.prev_theta = None
        self.omega_est = None
        self.omega_est_last = None
        self.v_est_last = None
        self.v_est = 0.0
        self.error_v_integral = 0
        self.error_v_last = 0
        self.idx = 0
        self.tracking_error = []

    def pose_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.theta = yaw

        if self.prev_x is None:
            self.prev_x = self.x
            self.prev_y = self.y
            self.prev_theta = self.theta
        alpha = np.exp(-2*np.pi*10*self.dt) # exp(w*T)
        # alpha = 0.001
        v_est_new = np.abs((np.sqrt(self.x**2+self.y**2) - np.sqrt(self.prev_x**2+self.prev_y**2))/self.dt)
        if self.v_est_last is None:
            self.v_est_last = v_est_new
        self.v_est = v_est_new*alpha + self.v_est_last*(1-alpha)
        self.v_est_last = self.v_est
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_theta = self.theta
            
    def pub_night_vapor(self):

        state_0 = np.array([self.x, self.y, self.theta, self.v_est])
        x_ref = xt[self.idx, 0]
        y_ref = xt[self.idx, 1]
        angle = xt[self.idx, 2]
        theta_ref = np.arctan2(np.sin(angle), np.cos(angle))

        # if (np.sign(theta_ref)!=np.sign(self.theta) and theta_ref<-2.8):
        #     # theta_ref= -theta_ref
        #     self.theta = -self.theta
        # elif(np.sign(theta_ref)!=np.sign(self.theta) and theta_ref>2.8):
        #     # theta_ref= -theta_ref
        #     self.theta = -self.theta


        # x = SE2.elem(ca.vertcat(self.x, self.y, self.theta))
        # xr = SE2.elem(ca.vertcat(x_ref, y_ref, theta_ref))
        # xi = (xr.inverse()*x).log()
        X_r = SE2(x_ref, y_ref, theta_ref)
        X = SE2(self.x, self.y, self.theta)
        eta = X_r.inv()@X
        xi = eta.log()

        self.tracking_error.append(np.sqrt((self.x - x_ref)**2 +(self.y - y_ref)**2)) 
        if (np.sqrt((self.x - xt[-1,0])**2 +(self.y - xt[-1, 1])**2)<1e-2):
            avg_track_error = np.mean(np.array(self.tracking_error))
            print('tracking error',avg_track_error) 

        if (float(xi.x) > -0.2) and (self.idx < len(ref_x_list)-1):
            self.idx += 1
        # elif (float(xi.param[0]) > -0.5) and (self.idx < len(ref_x_list)-1):
        #     self.idx += 1
        # print(float(xi.x), self.idx, X, X_r)
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
        a = float(cmd_vel[0])
        r_c = float(cmd_vel[1])

        K_th = 0.03 # high, faster, low, slower 0.1
        K_thi = 0.1
        K_thd = 0
        throttle_integral_max = 0.32
        ref_th = 0.16

        # Compute control commands for throttle and rudder
        # error_v = v - self.v_est
        # error_v_deriv = (error_v - self.error_v_last)/self.dt
        # self.error_v_last = error_v
        self.error_v_integral += xi.x*self.dt
        if self.error_v_integral > throttle_integral_max:
            self.error_v_integral = throttle_integral_max
        elif self.error_v_integral < -throttle_integral_max:
            self.error_v_integral = -throttle_integral_max
        self.throttle = 0.165*a + 0.15 #ref_th + 0.014 * a - 0.058 * xi.x + K_thi * self.error_v_integral #K_th*(error_v) + K_thi * self.error_v_integral + K_thd * error_v_deriv
        
        track_err = self.theta -theta_ref
        track_err = np.mod(track_err+np.pi,(2*np.pi)) - np.pi

        print(xt[self.idx,3], self.v_est, a, xi.x, theta_ref, self.theta, track_err)
        # Saturation for throttle
        if self.throttle < 0:
            self.throttle = 0
        if self.throttle > 1:
            self.throttle = 1

        delta = r_c * 6/np.pi
        # delta_integral = delta * dt
        # delta_integral_max = 0.4
        # if delta_integral > delta_integral_max:
        #     delta_integral = delta_integral_max
        # elif delta_integral < -delta_integral_max:
        #     delta_integral = -delta_integral_max

        # if (np.sign(theta_ref)!=np.sign(self.theta)):
        #     # theta_ref= -theta_ref
        #     delta = -delta
        # elif(np.sign(theta_ref)!=np.sign(self.theta)):
        #     # theta_ref= -theta_ref
        #     delta = -delta

        delta = delta #+ 0.05 * delta_integral
        print('MPC', delta)
        if delta < -1:
            delta = -1
        elif delta > 1:
            delta = 1
        else:
            delta = delta
        print('sat',self.idx, delta)

        joy_msg = Joy()

        joy_msg.axes = [0.0]*5
        # print(self.throttle)
        joy_msg.axes[0] = self.throttle
        joy_msg.axes[1] = delta
        # if self.time <= 3.5:
        #     joy_msg.axes[0] = 0.5 #1200 # throttle
        #     joy_msg.axes[1] = 0 #1500 # rudder
        # elif (3.5 < self.time <= 5.5):
        #     joy_msg.axes[0] = 0.6
        #     joy_msg.axes[1] = -0.5
        # elif (5.5< self.time <= 8.5):
        #     joy_msg.axes[0] = 0.5 #1200 # throttle
        #     joy_msg.axes[1] = 0 #1500 # rudder
        # else:
        #     joy_msg.axes[0] = 1
        joy_msg.axes[2] = 0
        joy_msg.axes[3] = 0
        joy_msg.axes[4] = 1900
       
        self.pub_joy.publish(joy_msg)

        
        # Append current position to the list
        self.x_list.append(self.x)
        self.y_list.append(self.y)
        self.z_list.append(self.z)
        self.theta_list.append(self.theta)
        #Publish the trajectory of the vehicle
        self.publish_path()

        msg_path = Path()
        msg_path.header.frame_id = 'qualisys'
        msg_path.header.stamp = self.get_clock().now().to_msg()
        for i in range(N):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'qualisys'
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
        ref_pos_msg.header.frame_id = 'qualisys'
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
        msg_path.header.frame_id = 'qualisys'
        msg_path.header.stamp = self.get_clock().now().to_msg()
        for x, y, theta in zip(ref_x_list, ref_y_list, ref_theta_list):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'qualisys'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = np.cos(theta/2)
            pose.pose.orientation.z = np.sin(theta/2)
            msg_path.poses.append(pose)
        self.pub_ref_path.publish(msg_path)

    def publish_path(self):
        msg_path = Path()
        msg_path.header.frame_id = 'qualisys'
        msg_path.header.stamp = self.get_clock().now().to_msg()
        if len(self.x_list) > self.trail_size:
            del self.x_list[0]
            del self.y_list[0]
            del self.theta_list[0]
        for x, y, theta in zip(self.x_list, self.y_list, self.theta_list):
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'qualisys'
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
