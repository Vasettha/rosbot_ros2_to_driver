import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
import serial
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np


class RobotDriver(Node):
    def __init__(self):
        super().__init__('robot_driver')
        
        # Configure serial connection
        try:
            self.serial_port = serial.Serial(
                port='/dev/ttyUSB0',
                baudrate=9600,
                timeout=0.1,
                writeTimeout=0.1
            )
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            raise e

        # Robot physical parameters
        self.WHEEL_SEPARATION = 0.23  # meters
        self.PULSES_PER_METER = 2343.0
        self.MAX_SPEED_PULSE_PER_LOOP = 25
        self.PID_LOOPS_PER_SECOND = 40

        # QoS profile for odometry
        odom_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # Publishers, subscribers and broadcasters
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )
        self.odom_pub = self.create_publisher(
            Odometry,
            'odom',
            odom_qos
        )
        self.odom_broadcaster = TransformBroadcaster(self)

        # Robot state variables
        self.pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.twist = {'linear': 0.0, 'angular': 0.0}
        self.last_encoder_readings = {'left': 0.0, 'right': 0.0}
        self.current_time = self.get_clock().now()
        self.last_time = self.current_time

        # Create timers
        self.create_timer(0.02, self.update_odometry)  # 50Hz for odometry updates
        self.create_timer(0.1, self.watchdog_timer)    # 10Hz for watchdog
        
        # Watchdog for cmd_vel
        self.last_cmd_time = self.get_clock().now()
        self.CMD_TIMEOUT = 0.5  # seconds
        
        self.get_logger().info('Robot driver initialized')

    def cmd_vel_callback(self, msg):
        """
        Handle incoming velocity commands.
        Converts linear and angular velocities to wheel speeds.
        """
        try:
            # Update command timestamp
            self.last_cmd_time = self.get_clock().now()
            
            # Convert velocity commands to wheel speeds
            linear_speed = msg.linear.x * self.PULSES_PER_METER / self.PID_LOOPS_PER_SECOND
            angular_speed = msg.angular.z * self.WHEEL_SEPARATION / 2 * self.PULSES_PER_METER / self.PID_LOOPS_PER_SECOND

            # Calculate individual wheel speeds
            left_speed = int(linear_speed - angular_speed)
            right_speed = int(linear_speed + angular_speed)

            # Clamp speeds to maximum allowed values
            left_speed = np.clip(left_speed, -self.MAX_SPEED_PULSE_PER_LOOP, self.MAX_SPEED_PULSE_PER_LOOP)
            right_speed = np.clip(right_speed, -self.MAX_SPEED_PULSE_PER_LOOP, self.MAX_SPEED_PULSE_PER_LOOP)

            # Send command to robot
            command = f"m {left_speed} {right_speed}\n"
            self.serial_port.write(command.encode())
            self.serial_port.flush()

        except serial.SerialException as e:
            self.get_logger().error(f'Serial communication error in cmd_vel: {e}')
        except Exception as e:
            self.get_logger().error(f'Error in cmd_vel callback: {e}')

    def read_encoders(self):
        """
        Request and read encoder values from the robot.
        Returns tuple of (left_mm, right_mm) or None if error occurs.
        """
        try:
            # Clear any pending data
            self.serial_port.reset_input_buffer()
            
            # Send encoder read command
            self.serial_port.write(b"e\n")
            self.serial_port.flush()
            
            # Read response
            response = self.serial_port.readline().decode().strip()
            
            if not response:
                return None
                
            # Parse encoder values
            left_str, right_str = response.split("Right encoder (mm):")
            left_mm = float(left_str.split("Left encoder (mm):")[1].strip())
            right_mm = float(right_str.strip())
            
            return (left_mm, right_mm)
            
        except (serial.SerialException, ValueError, IndexError) as e:
            self.get_logger().error(f'Error reading encoders: {e}')
            return None

    def update_odometry(self):
        """
        Update robot odometry based on encoder readings.
        Publishes odometry data and broadcasts transforms.
        """
        try:
            # Read encoder values
            encoder_readings = self.read_encoders()
            if encoder_readings is None:
                return

            # Convert mm to meters
            left_m = encoder_readings[0] / 1000.0
            right_m = encoder_readings[1] / 1000.0

            # Calculate wheel travel distances since last update
            d_left = left_m - self.last_encoder_readings['left']
            d_right = right_m - self.last_encoder_readings['right']

            # Update stored encoder readings
            self.last_encoder_readings['left'] = left_m
            self.last_encoder_readings['right'] = right_m

            # Calculate robot movement
            d_center = (d_right + d_left) / 2.0
            d_theta = (d_right - d_left) / self.WHEEL_SEPARATION

            # Update robot pose
            self.pose['theta'] = (self.pose['theta'] + d_theta) % (2 * math.pi)
            self.pose['x'] += d_center * math.cos(self.pose['theta'])
            self.pose['y'] += d_center * math.sin(self.pose['theta'])

            # Get current time and time difference
            self.current_time = self.get_clock().now()
            dt = (self.current_time - self.last_time).nanoseconds / 1e9

            if dt > 0:
                self.twist['linear'] = d_center / dt
                self.twist['angular'] = d_theta / dt

            # Create and publish odometry message
            odom_msg = Odometry()
            odom_msg.header.stamp = self.current_time.to_msg()
            odom_msg.header.frame_id = 'odom'
            odom_msg.child_frame_id = 'base_link'

            # Set position
            odom_msg.pose.pose.position.x = self.pose['x']
            odom_msg.pose.pose.position.y = self.pose['y']
            odom_msg.pose.pose.position.z = 0.0

            # Set orientation (as quaternion)
            odom_msg.pose.pose.orientation.z = math.sin(self.pose['theta'] / 2.0)
            odom_msg.pose.pose.orientation.w = math.cos(self.pose['theta'] / 2.0)

            # Set velocities
            odom_msg.twist.twist.linear.x = self.twist['linear']
            odom_msg.twist.twist.angular.z = self.twist['angular']

            # Publish odometry message
            self.odom_pub.publish(odom_msg)

            # Broadcast transform
            transform_msg = TransformStamped()
            transform_msg.header.stamp = self.current_time.to_msg()
            transform_msg.header.frame_id = 'odom'
            transform_msg.child_frame_id = 'base_link'
            
            transform_msg.transform.translation.x = self.pose['x']
            transform_msg.transform.translation.y = self.pose['y']
            transform_msg.transform.translation.z = 0.0
            
            transform_msg.transform.rotation.z = math.sin(self.pose['theta'] / 2.0)
            transform_msg.transform.rotation.w = math.cos(self.pose['theta'] / 2.0)

            self.odom_broadcaster.sendTransform(transform_msg)

            # Update last time
            self.last_time = self.current_time

        except Exception as e:
            self.get_logger().error(f'Error updating odometry: {e}')

    def watchdog_timer(self):
        """
        Monitor command velocity timeout and stop robot if necessary.
        """
        try:
            if (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9 > self.CMD_TIMEOUT:
                # Stop the robot
                self.serial_port.write(b"s\n")
                self.serial_port.flush()
        except serial.SerialException as e:
            self.get_logger().error(f'Serial error in watchdog: {e}')

    def cleanup(self):
        """
        Clean up resources before node shutdown.
        """
        try:
            # Stop the robot
            self.serial_port.write(b"s\n")
            self.serial_port.flush()
            # Close serial port
            self.serial_port.close()
        except serial.SerialException as e:
            self.get_logger().error(f'Error during cleanup: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    driver = RobotDriver()
    
    try:
        rclpy.spin(driver)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        driver.get_logger().error(f'Unexpected error: {e}')
    finally:
        driver.cleanup()
        driver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()