# Group Number 
VOTRE_NUMERO_EQUIPE = 7

# Import required packages and configure the ROS Master
import numpy as np
import rospy
import time 
from scipy.spatial.transform import Rotation as R
from math import sin, cos
from jackal_msgs.msg import Feedback
from jackal_msgs.msg import Drive
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from filterpy.kalman import KalmanFilter

# Create and start a new node
rospy.init_node('dingo_state_estimation', anonymous=True)
rate = rospy.Rate(50) # Update frequency in Hz

# SCRIPT CONFIGURATION 
# Variables with the robot's dimensions in [m]
interwheel_distance = 0.3765
left_wheel_radius = 0.049
right_wheel_radius = 0.049

# Function to calculate orientation from a quaternion
# Input: Quaternion [x, y, z, w]
# Output: Yaw angle (heading) in radians
def get_heading_from_quaternion(q):
    r = R.from_quat([q.x, q.y, q.z, q.w])
    angles = r.as_euler('xyz', degrees=False)
    return angles[2]

# Ground truth subscriber callback
ground_truth_msg = Odometry()
def ground_truth_callback(msg):
    global ground_truth_msg
    ground_truth_msg = msg

# IMU subscriber callback 
acc_x_imu = 0
acc_y_imu = 0
vang_imu = 0
acc_lin_imu = 0
heading_imu = 0
imu_msg = Imu()
def imu_callback(msg):
    # KF update: fill the matrix with variables retrieved from the IMU
    global acc_x_imu, acc_y_imu,vang_imu, acc_lin_imu, heading_imu, imu_msg
    imu_msg = msg

    # Acceleration along the x and y axes in [m/s^2]
    acc_x_imu = msg.linear_acceleration.x
    acc_y_imu = msg.linear_acceleration.y

    # Calculation of the resultant linear acceleration in [m/s^2]
    acc_lin_imu = np.sqrt(acc_x_imu**2 + acc_y_imu**2)

    # Heading in [rad]
    heading_imu = get_heading_from_quaternion(msg.orientation)

    # Angular velocity in [rad/s]
    vang_imu = msg.angular_velocity.z

    kf_imu_update(np.array([acc_x_imu, acc_y_imu, heading_imu, vang_imu]), rospy.get_time())

# Encoders subscriber callback 
prev_left_pos = 0.0
prev_right_pos = 0.0
vit_lin = 0.0
vit_ang = 0.0
last_time = rospy.Time.now()
enc_msg = Feedback()
def encoders_callback(msg):
    global enc_msg, prev_left_pos, prev_right_pos, vit_lin, vit_ang, last_time, kf_u
    enc_msg = msg

    # Get dt
    time = rospy.Time.now()
    dt = (time - last_time).to_sec()
    if dt < 0.0001:
        return
    last_time = time

    # Encoder measurements in [rad]
    left_pos = msg.drivers[0].measured_travel
    right_pos = msg.drivers[1].measured_travel

    # Calculate the measured wheel speeds in [rad/s]
    left_vel = (left_pos - prev_left_pos) / dt 
    right_vel = (right_pos - prev_right_pos) / dt

    # Calculate the linear speed of the wheels in [m/s]
    v_left = left_wheel_radius * left_vel
    v_right = right_wheel_radius * right_vel

    # Calculate the linear and angular speed of the robot in [m/s] and [rad/s]
    vit_lin = (v_left + v_right) / 2
    vit_ang = (v_right - v_left) / interwheel_distance
    
    # Update the control input matrix if using a control-based KF
    kf_u = np.array([vit_lin, vit_ang])

    # Update variables
    prev_left_pos = left_pos
    prev_right_pos = right_pos

# ROS subscribers and publishers 
feedback_sub = rospy.Subscriber('/mobile_manip/dingo_velocity_controller/feedback', Feedback, encoders_callback)
imu_sub = rospy.Subscriber('/imu/data', Imu, imu_callback)
ground_truth_sub = rospy.Subscriber('/mobile_manip/ground_truth/state', Odometry, ground_truth_callback)
cmd_drive_pub = rospy.Publisher('/mobile_manip/dingo_velocity_controller/cmd_drive', Drive, queue_size=1)

# KALMAN FILTER CONSTRUCTION 
# Create the Kalman filter
# dim_x: states
# dim_z: measurements
# dim_u: controls
kf = KalmanFilter(dim_x=9, dim_z=4, dim_u=2)

# Update frequency 
dt = 1/50

# Initial state (adjust the matrix size as needed)
kf.x = np.zeros(9)
# STATE ESTIMATOR 
x = kf.x[0]
vit_x = kf.x[1]
acc_x = kf.x[2]
y = kf.x[3]
vit_y = kf.x[4]
acc_y = kf.x[5]
theta = kf.x[6]
vit_ang = kf.x[7]
acc_ang = kf.x[8]

# Control-state transition matrix if using a control-based KF 
kf.B = np.array([[dt*cos(theta), 0],
                 [cos(theta), 0],
                 [0, 0],
                 [dt*sin(theta), 0],
                 [sin(theta), 0],
                 [0, 0],
                 [0, dt],
                 [0, 0],
                 [0, 0]])
                
# Process noise matrix 
var_vg_enc = 0.002072 # Variance of the left wheel's linear speed
var_vd_enc = 0.001964 # Variance of the right wheel's linear speed
var_vlin_enc = (var_vg_enc + var_vd_enc)/4 # Variance of the robot's linear speed
var_vang_enc = (var_vd_enc + var_vg_enc)/interwheel_distance**2 # Variance of the robot's angular speed

kf_Q = np.array([[var_vlin_enc, 0],
                 [0, var_vang_enc]])

Q_temp = np.dot(kf.B, kf_Q)
kf.Q = np.dot(Q_temp, kf.B.T)

# Initialize the covariance matrix
kf.P *= 500 # The library initializes the matrix as an identity matrix, just multiply by an arbitrary factor to initialize the first instance

# IMU noise matrix
var_acc_x_imu = 0.000024  # Variance of IMU's acceleration along the x-axis
var_acc_y_imu = 0.000025  # Variance of IMU's acceleration along the y-axis
var_heading_imu = 0.000025  # Variance of IMU's heading
var_angular_vel_imu = 0.000024 # Variance of IMU's angular velocity

R_imu = np.array([[var_acc_x_imu, 0, 0, 0],
                  [0, var_acc_y_imu, 0, 0],
                  [0, 0, var_heading_imu, 0],
                  [0, 0, 0, var_angular_vel_imu]])

# IMU-state transition matrix
acc_x_inertial = acc_x * cos(theta) - acc_y * sin(theta)
acc_y_inertial = acc_x * sin(theta) + acc_y * cos(theta)
acc_resultant_inertial = np.sqrt(acc_x_inertial**2 + acc_y_inertial**2)
if acc_resultant_inertial == 0:
    acc_resultant_inertial = 0.00001

H_imu = np.array([[0, 0, cos(theta), 0, 0 , -sin(theta), -acc_x*sin(theta)-acc_y*cos(theta), 0, 0],
                  [0, 0, sin(theta), 0, 0, cos(theta), acc_x*cos(theta)-acc_y*sin(theta), 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0]])

# Encoder noise matrix 
R_enc = np.zeros((2, 2))
# Encoder-state transition matrix 
H_enc = np.zeros((2, 9))

# Initialize the control input matrix if using a control-based KF 
kf_u = np.zeros(2)

# Model-state transition matrix 
kf.F = np.array([[1, 0, 1/2*(dt**2), 0, 0, 0, 0, 0, 0],
                 [0, 0, dt, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 1, 0, 1/2*(dt**2), 0, 0, 0],
                 [0, 0, 0, 0, 0, dt, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 1/2*(dt**2)],
                 [0, 0, 0, 0, 0, 0, 0, 1, dt],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1]])

# Kalman filter prediction function
last_measurement_time = 0
def kf_pre_update(timestamp):
    global last_measurement_time, H_imu

    if timestamp > last_measurement_time:
        dt = timestamp - last_measurement_time
        last_measurement_time = timestamp

        acc_x = kf.x[2]
        acc_y = kf.x[5]
        theta = kf.x[6]
        vlin = vit_lin

        # Model-state transition matrix 
        kf.F = np.array([[1, 0, 1/2*(dt**2), 0, 0, 0, 0, 0, 0],
                 [0, 0, dt, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 0], 
                 [0, 0, 0, 1, 0, 1/2*(dt**2), 0, 0, 0],
                 [0, 0, 0, 0, 0, dt, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 1/2*(dt**2)],
                 [0, 0, 0, 0, 0, 0, 0, 1, dt],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        
        # Control-state transition matrix if using a control-based KF 
        kf.B = np.array([[dt*cos(theta), 0],
                 [cos(theta), 0],
                 [0, 0],
                 [dt*sin(theta), 0],
                 [sin(theta), 0],
                 [0, 0],
                 [0, dt],
                 [0, 0],
                 [0, 0]])

        # Update the IMU-state transition matrix 
        H_imu = np.array([[0, 0, cos(theta), 0, 0 , -sin(theta), -acc_x*sin(theta)-acc_y*cos(theta), 0, 0],
                  [0, 0, sin(theta), 0, 0, cos(theta), acc_x*cos(theta)-acc_y*sin(theta), 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0]])

    else:
        pass

    # Prediction function
    kf.predict(u=kf_u) # Used if using a control-based KF
    #kf.predict() # Used if not using a control-based KF

# Function to update IMU measurements
def kf_imu_update(z, timestamp):
    # IMU-related matrices (noise, transition)
    kf.R = R_imu
    kf.H = H_imu
    # Call the prediction function
    kf_pre_update(timestamp)
    kf.update(z)

# Function to update encoder measurements
def kf_enc_update(z, timestamp):
    # Encoder-related matrices (noise, transition)
    kf.R = R_enc
    kf.H = H_enc
    # Call the prediction function
    kf_pre_update(timestamp)
    kf.update(z)

# TESTING THE ESTIMATION
# Move the robot along a curve and record a rosbag
# Create a new ROSBAG for writing
import rosbag
test_bag = rosbag.Bag('kalman_filter_9_states.bag', 'w')

# Publish odometry and TF for 40 seconds
start = float(rospy.Time().now().secs)
rate = rospy.Rate(50) # 50Hz
while (float(rospy.Time().now().secs) - start) < 40:
    
    x = kf.x[0]
    vit_x = kf.x[1]
    acc_x = kf.x[2]
    y = kf.x[3]
    vit_y = kf.x[4]
    acc_y = kf.x[5]
    theta = kf.x[6]
    vit_ang = kf.x[7]
    acc_ang = kf.x[8]
    vit_resultante = np.sqrt(vit_x**2 + vit_y**2)

    # Move the robot
    cmd_drive_msg = Drive()
    cmd_drive_msg.drivers[0] = 5.0
    cmd_drive_msg.drivers[1] = 7.0
    cmd_drive_pub.publish(cmd_drive_msg)
    
    # Record the robot's movement
    pose = (x, y, 0)
    r = R.from_euler('xyz', [0, 0, theta], degrees=False)
    orientation = r.as_quat()
    odometry_msg = Odometry()
    odometry_msg.header.frame_id = "odom"
    odometry_msg.header.stamp = rospy.Time.now()
    odometry_msg.child_frame_id = "base_link"
    odometry_msg.pose.pose = Pose(Point(*pose), Quaternion(*orientation))
    odometry_msg.twist.twist = Twist(Vector3(vit_resultante, 0, 0), Vector3(0, 0, vit_ang))
    
    test_bag.write('/raw_enc', enc_msg, rospy.Time().now())
    test_bag.write('/raw_imu', imu_msg, rospy.Time().now())   
    test_bag.write('/filter', odometry_msg, rospy.Time().now())
    test_bag.write('/ground_truth', ground_truth_msg, rospy.Time().now())
    rate.sleep()
    
# Properly close the ROSBAG
test_bag.close()

# Stop the Dingo robot
cmd_drive_msg = Drive()
cmd_drive_msg.drivers[0] = 0.0
cmd_drive_msg.drivers[1] = 0.0
cmd_drive_pub.publish(cmd_drive_msg)
