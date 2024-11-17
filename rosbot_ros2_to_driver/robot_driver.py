import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from example_interfaces.srv import Trigger
import serial
import math

class RobotDriver(Node):
    def __init__(self):
        super().__init__('robot_driver')
        self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # Match ESP32 baud rate

        # Robot-specific constants
        self.PULSES_PER_METER = 2343
        self.PID_LOOPS_PER_SECOND = 40
        self.MAX_SPEED_PULSE_PER_LOOP = 25
        self.WHEEL_SEPARATION = 0.23

        # Create publishers and subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        
        # Create the odometry service
        self.get_odom_srv = self.create_service(
            Trigger, 
            'get_odometry', 
            self.get_odometry_callback
        )

        self.odom_broadcaster = TransformBroadcaster(self)

        # Variables to store robot's position
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.prev_left_distance = 0.0
        self.prev_right_distance = 0.0
        self.last_encoder_time = self.get_clock().now()
        
        # Request encoder data more frequently
        self.encoder_timer = self.create_timer(0.05, self.request_encoder_data)
        self.odom_timer = self.create_timer(0.05, self.publish_odometry)

    def request_encoder_data(self):
        """Request encoder data from ESP32"""
        try:
            self.serial_port.write(b"e\n")
            serial_response = self.serial_port.readline().decode().strip()
            
            if not serial_response:
                self.get_logger().warn("No response from serial port")
                return
                
            # Parse the response
            left_str, right_str = serial_response.split("Right encoder (mm):")
            left_distance = float(left_str.split("Left encoder (mm):")[1].strip())
            right_distance = float(right_str.strip())

            # Convert mm to meters
            left_distance_m = left_distance / 1000.0
            right_distance_m = right_distance / 1000.0

            # Calculate change in distance
            delta_left = left_distance_m - self.prev_left_distance
            delta_right = right_distance_m - self.prev_right_distance

            # Update previous distances
            self.prev_left_distance = left_distance_m
            self.prev_right_distance = right_distance_m

            # Get time difference
            current_time = self.get_clock().now()
            dt = (current_time - self.last_encoder_time).nanoseconds / 1e9
            self.last_encoder_time = current_time

            if dt > 0:
                # Calculate robot's movement
                delta_distance = (delta_right + delta_left) / 2
                delta_theta = (delta_right - delta_left) / self.WHEEL_SEPARATION

                # Update robot's position and orientation
                self.x += delta_distance * math.cos(self.theta + delta_theta / 2)
                self.y += delta_distance * math.sin(self.theta + delta_theta / 2)
                self.theta = (self.theta + delta_theta) % (2 * math.pi)

        except Exception as e:
            self.get_logger().error(f"Error reading encoder data: {e}")

    def publish_odometry(self):
        current_time = self.get_clock().now()
        
        # Create and populate the odometry message
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        # Set the position
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0

        # Convert theta to quaternion
        odom.pose.pose.orientation.z = math.sin(self.theta / 2)
        odom.pose.pose.orientation.w = math.cos(self.theta / 2)

        # Calculate velocities
        dt = (current_time - self.last_encoder_time).nanoseconds / 1e9
        if dt > 0:
            linear_velocity = (self.prev_right_distance + self.prev_left_distance) / (2 * dt)
            angular_velocity = (self.prev_right_distance - self.prev_left_distance) / (self.WHEEL_SEPARATION * dt)
        else:
            linear_velocity = 0.0
            angular_velocity = 0.0

        odom.twist.twist.linear.x = linear_velocity
        odom.twist.twist.angular.z = angular_velocity

        # Publish the odometry message
        self.odom_pub.publish(odom)

        # Create and send the transform
        odom_trans = TransformStamped()
        odom_trans.header.stamp = current_time.to_msg()
        odom_trans.header.frame_id = 'odom'
        odom_trans.child_frame_id = 'base_link'
        odom_trans.transform.translation.x = self.x
        odom_trans.transform.translation.y = self.y
        odom_trans.transform.translation.z = 0.0
        odom_trans.transform.rotation.z = math.sin(self.theta / 2)
        odom_trans.transform.rotation.w = math.cos(self.theta / 2)

        self.odom_broadcaster.sendTransform(odom_trans)

    # ... rest of the class implementation remains the same ...
