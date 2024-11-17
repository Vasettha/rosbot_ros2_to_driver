import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import serial
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
from threading import Lock

class RobotDriver(Node):
    def __init__(self):
        super().__init__('robot_driver')
        
        # Serial communication lock
        self.serial_lock = Lock()
        
        # Configure serial connection
        try:
            self.serial_port = serial.Serial(
                port='/dev/ttyUSB1',
                baudrate=115200,
                timeout=0.1,
                writeTimeout=0.1
            )
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            raise e

        # Robot physical parameters
        self.WHEEL_SEPARATION = 0.23  # meters
        self.WHEEL_RADIUS = 0.0335    # meters
        self.PULSES_PER_METER = 2343.0
        self.MAX_SPEED_PULSE_PER_LOOP = 25
        self.PID_LOOPS_PER_SECOND = 40

        # QoS profile for odometry
        odom_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        # Publishers and subscribers
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
        self.joint_pub = self.create_publisher(
            JointState,
            'joint_states',
            10
        )
        self.odom_broadcaster = TransformBroadcaster(self)

        # Robot state variables
        self.pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.twist = {'linear': 0.0, 'angular': 0.0}
        self.last_encoder_readings = {'left': 0.0, 'right': 0.0}
        self.wheel_positions = {'left': 0.0, 'right': 0.0}
        self.current_time = self.get_clock().now()
        self.last_time = self.current_time
        
        # Command processing variables
        self.last_cmd_time = self.get_clock().now()
        self.CMD_TIMEOUT = 0.5  # seconds
        self.last_cmd = (0, 0)  # Store last commanded speeds
        
        # Create timers
        self.create_timer(0.02, self.update_odometry)  # 50Hz for odometry
        self.create_timer(0.1, self.watchdog_timer)    # 10Hz for watchdog
        
        self.get_logger().info('Robot driver initialized')
        
    def send_serial_command(self, command):
        """Thread-safe serial command sending with retry logic"""
        try:
            with self.serial_lock:
                # Only clear input buffer if it's getting full
                if self.serial_port.in_waiting > 64:
                    self.serial_port.reset_input_buffer()
                
                self.serial_port.write(command.encode())
                self.serial_port.flush()
                
                # Wait for confirmation from ESP32
                response = self.serial_port.readline().decode().strip()
                if not response:
                    self.get_logger().warn('No response from robot')
                    return False
                    
                return True
                
        except serial.SerialException as e:
            self.get_logger().error(f'Serial error sending command: {e}')
            return False
            
    def cmd_vel_callback(self, msg):
        """Handle incoming velocity commands"""
        try:
            # Update command timestamp
            self.last_cmd_time = self.get_clock().now()
            
            # Convert velocity commands to wheel speeds
            linear_speed = msg.linear.x * self.PULSES_PER_METER / self.PID_LOOPS_PER_SECOND
            angular_speed = msg.angular.z * self.WHEEL_SEPARATION / 2 * self.PULSES_PER_METER / self.PID_LOOPS_PER_SECOND
    
            # Calculate individual wheel speeds
            left_speed = int(linear_speed - angular_speed)
            right_speed = int(linear_speed + angular_speed)
    
            # Clamp speeds
            left_speed = np.clip(left_speed, -self.MAX_SPEED_PULSE_PER_LOOP, self.MAX_SPEED_PULSE_PER_LOOP)
            right_speed = np.clip(right_speed, -self.MAX_SPEED_PULSE_PER_LOOP, self.MAX_SPEED_PULSE_PER_LOOP)
            
            # Only send command if speeds have changed
            if (left_speed, right_speed) != self.last_cmd:
                command = f"m {left_speed} {right_speed}\n"
                if self.send_serial_command(command):
                    self.last_cmd = (left_speed, right_speed)
    
        except Exception as e:
            self.get_logger().error(f'Error in cmd_vel callback: {e}')

    def read_encoders(self):
        """Thread-safe encoder reading with improved error handling"""
        try:
            with self.serial_lock:
                self.serial_port.write(b"e\n")
                self.serial_port.flush()
                
                response = self.serial_port.readline().decode().strip()
                
                if not response:
                    return None
                    
                if "Left encoder (mm):" in response and "Right encoder (mm):" in response:
                    try:
                        # Split and parse with more robust error handling
                        parts = response.split("Right encoder")
                        left_part = parts[0].split("Left encoder (mm):")
                        right_part = parts[1].split(":")
                        
                        left_mm = float(left_part[1].strip())
                        right_mm = float(right_part[1].strip())
                        
                        return (left_mm, right_mm)
                        
                    except (IndexError, ValueError) as e:
                        self.get_logger().debug(f'Error parsing encoder values: {e}')
                        return None
                else:
                    self.get_logger().debug(f'Unexpected response format: {response}')
                    return None
                    
        except serial.SerialException as e:
            self.get_logger().error(f'Serial error reading encoders: {e}')
            return None

    def publish_joint_states(self, d_left, d_right, dt):
        """Publish joint states for the wheels"""
        try:
            # Calculate wheel positions (accumulated angle)
            self.wheel_positions['left'] += d_left / self.WHEEL_RADIUS
            self.wheel_positions['right'] += d_right / self.WHEEL_RADIUS

            # Calculate wheel velocities
            left_velocity = d_left / (dt * self.WHEEL_RADIUS) if dt > 0 else 0
            right_velocity = d_right / (dt * self.WHEEL_RADIUS) if dt > 0 else 0

            # Create joint state message
            joint_state = JointState()
            joint_state.header.stamp = self.current_time.to_msg()
            joint_state.name = ['left_wheel_joint', 'right_wheel_joint']
            joint_state.position = [self.wheel_positions['left'], self.wheel_positions['right']]
            joint_state.velocity = [left_velocity, right_velocity]
            joint_state.effort = []

            # Publish joint states
            self.joint_pub.publish(joint_state)

        except Exception as e:
            self.get_logger().error(f'Error publishing joint states: {e}')

    def update_odometry(self):
        """Update robot odometry based on encoder readings"""
        try:
            # Read encoder values
            encoder_readings = self.read_encoders()
            if encoder_readings is None:
                self.get_logger().debug('Failed to read encoders, skipping odometry update')
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

            # Get current time and time difference
            self.current_time = self.get_clock().now()
            dt = (self.current_time - self.last_time).nanoseconds / 1e9

            # Only proceed if we have a valid time difference
            if dt > 0:
                # Publish joint states
                self.publish_joint_states(d_left, d_right, dt)

                # Update robot pose
                self.pose['theta'] = (self.pose['theta'] + d_theta) % (2 * math.pi)
                self.pose['x'] += d_center * math.cos(self.pose['theta'])
                self.pose['y'] += d_center * math.sin(self.pose['theta'])

                # Update twist
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

                # Set orientation (using theta for simple 2D rotation)
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
        """Monitor command velocity timeout"""
        try:
            if (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9 > self.CMD_TIMEOUT:
                if self.last_cmd != (0, 0):  # Only send stop if we're not already stopped
                    self.send_serial_command("s\n")
                    self.last_cmd = (0, 0)
                    
        except Exception as e:
            self.get_logger().error(f'Error in watchdog: {e}')

    def cleanup(self):
        """Clean up resources before shutdown"""
        try:
            self.send_serial_command("s\n")
            self.serial_port.close()
        except Exception as e:
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
