#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Twist, Pose
import tf
import numpy as np
import transformations

def quaternion_to_euler_angles(qx, qy, qz, qw):
    # Convert quaternion to numpy array
    q_array = np.array([qx, qy, qz, qw])

    # Compute Euler angles (roll, pitch, yaw)
    euler_angles = transformations.euler_from_quaternion(q_array)

    return euler_angles[0], euler_angles[1], euler_angles[2]


def get_odom_from_estimated_pose(new_pose_line, last_pose_line):
    timestamp = new_pose_line[0]
    x = new_pose_line[1]
    y = new_pose_line[2]
    z = new_pose_line[3]
    qx = new_pose_line[4]   
    qy = new_pose_line[5]
    qz = new_pose_line[6]
    qw = new_pose_line[7]
    alpha, beta, theta = quaternion_to_euler_angles(qx, qy, qz, qw)

    timestamp_last = last_pose_line[0]
    x_last = last_pose_line[1]
    y_last = last_pose_line[2]
    z_last = last_pose_line[3]
    qx_last = last_pose_line[4]
    qy_last = last_pose_line[5]
    qz_last = last_pose_line[6]
    qw_last = last_pose_line[7]
    alpha_last, beta_last, theta_last = quaternion_to_euler_angles(qx_last, qy_last, qz_last, qw_last)

    dt = timestamp - timestamp_last
    
    x_delta = x - x_last
    y_delta = y - y_last
    z_delta = z - z_last
    alpha_delta = alpha - alpha_last
    beta_delta = beta - beta_last
    theta_delta = theta - theta_last

    linear_velocity_x = x_delta / dt
    linear_velocity_y = y_delta / dt
    linear_velocity_z = z_delta / dt
    angular_velocity_alpha = alpha_delta / dt
    angular_velocity_beta = beta_delta / dt
    angular_velocity_theta = theta_delta / dt


    odom_msg = Odometry()
    odom_msg.header.stamp = rospy.Time.from_sec(timestamp)
    odom_msg.header.frame_id = "odom"
    odom_msg.child_frame_id = "body"

    odom_msg.pose.pose = Pose()
    odom_msg.pose.pose.position = Point(x, y, z)  
    odom_msg.pose.pose.orientation = Quaternion(qx, qy, qz, qw) 
    
    odom_msg.twist.twist = Twist()
    odom_msg.twist.twist.linear.x = linear_velocity_x
    odom_msg.twist.twist.linear.y = linear_velocity_y
    odom_msg.twist.twist.linear.z = linear_velocity_z
    odom_msg.twist.twist.angular.x = angular_velocity_alpha
    odom_msg.twist.twist.angular.y = angular_velocity_beta
    odom_msg.twist.twist.angular.z = angular_velocity_theta
    
    return odom_msg


def publish_odometry_from_file():
    rospy.init_node('irmcl_odometry_publisher', anonymous=True)
    file_path = rospy.get_param('file_path', default='../results/simulation/SE1/T1/IRMCL.txt')
    topic_name = rospy.get_param('topic_name', default='/irmcl/odometry')
    rate = rospy.Rate(10) 

    odom_pub = rospy.Publisher(topic_name, Odometry, queue_size=10)

    last_pose_line = np.zeros(8)
    rate.sleep()

    with open(file_path, 'r') as file:
        for line in file:
            data = line.split()
 
            t, x, y, z, qx, qy, qz, qw = map(float, data)
            new_pose_line = np.array([t, x, y, z, qx, qy, qz, qw])

            odom_msg = get_odom_from_estimated_pose(new_pose_line, last_pose_line)

            odom_pub.publish(odom_msg)

            rate.sleep()


if __name__ == '__main__':
    try:
        publish_odometry_from_file()
    except rospy.ROSInterruptException:
        pass
