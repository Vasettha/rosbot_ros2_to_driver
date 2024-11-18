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
import time

class RobotDriver(Node):
    def __init__(self):
        super().__init__('robot_driver')
        
        # Robot physical parameters
        self.WHEEL_SEPARATION = 0.23  # meters
        self.WHEEL_RADIUS = 0.0335    # meters
        self.PULSES_PER_METER = 2343.0
        self.MAX_SPEED_PULSE_PER_LOOP = 25  # Maximum pulses per PID loop (40Hz)
        self.PID_LOOPS_PER_SECOND = 40
        self.SERIAL_TIMEOUT = 0.05    # 50ms timeout for serial operations
        
        # Configure QoS
        odom_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        
        # Initialize serial with mutex
        self.serial_mutex = Lock()
        self.init_serial()
        
        # Publishers and subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, 'odom', odom_qos)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Robot state
        self.pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.twist = {'linear': 0.0, 'angular': 0.0}
        self.wheel_positions = {'left': 0.0, 'right': 0.0}
        self.last_encoder_readings = {'left': 0.0, 'right': 0.0}
        
        # Timing management
        self.last_cmd_time = self.get_clock().now()
        self.last_odom_time = self.get_clock().now()
        self.CMD_TIMEOUT = 0.5  # seconds
        
        # Initialize encoders
        self.reset_encoders()
        
        # Create timers
        self.create_timer(0.02, self.update_odometry)  # 50Hz odometry
        self.create_timer(0.1, self.watchdog_timer)    # 10Hz watchdog
        
        self.get_logger().info('Robot driver initialized')

    def init_serial(self):
        """Initialize serial connection with retry mechanism."""
        while True:
            try:
                self.serial_port = serial.Serial(
                    port='/dev/ttyUSB1',
                    baudrate=115200,
                    timeout=self.SERIAL_TIMEOUT,
                    write_timeout=self.SERIAL_TIMEOUT
                )
                break
            except serial.SerialException as e:
                self.get_logger().error(f'Failed to open serial port: {e}')
                time.sleep(1.0)  # Wait before retry

    def send_command(self, command, expect_response=True):
        """Thread-safe serial command sending with error handling."""
        try:
            with self.serial_mutex:
                # Clear input buffer
                self.serial_port.reset_input_buffer()
                
                # Send command
                self.serial_port.write(command.encode())
                self.serial_port.flush()
                
                if expect_response:
                    response = self.serial_port.readline().decode().strip()
                    return response
                return None
                
        except serial.SerialException as e:
            self.get_logger().error(f'Serial error: {e}')
            self.init_serial()  # Attempt to reconnect
            return None

    def reset_encoders(self):
        """Reset encoder values."""
        response = self.send_command('r\n')
        if response != 'OK':
            self.get_logger().warn('Failed to reset encoders')
            
    def read_encoders(self):
        """Read encoder values with error handling."""
        response = self.send_command('e\n')
        if response:
            try:
                left_mm, right_mm = map(float, response.split())
                return left_mm / 1000.0, right_mm / 1000.0  # Convert to meters
            except (ValueError, IndexError) as e:
                self.get_logger().error(f'Error parsing encoder values: {e}')
        return None

    def cmd_vel_callback(self, msg):
        """Handle incoming velocity commands with rate limiting."""
        try:
            # Update command timestamp
            current_time = self.get_clock().now()
            dt = (current_time - self.last_cmd_time).nanoseconds / 1e9
            
            # Rate limiting to 10Hz
            if dt < 0.1:
                return
                
            self.last_cmd_time = current_time
            
            # Convert velocity commands to wheel speeds
            linear_speed = msg.linear.x
            angular_speed = msg.angular.z
            
            # Calculate required pulses per PID loop
            linear_pulses = linear_speed * self.PULSES_PER_METER / self.PID_LOOPS_PER_SECOND
            angular_pulses = angular_speed * self.WHEEL_SEPARATION / 2.0 * \
                           self.PULSES_PER_METER / self.PID_LOOPS_PER_SECOND
            
            # Calculate individual wheel speeds
            left_pulses = int(linear_pulses - angular_pulses)
            right_pulses = int(linear_pulses + angular_pulses)
            
            # Clamp values
            left_pulses = np.clip(left_pulses, -self.MAX_SPEED_PULSE_PER_LOOP,
                                self.MAX_SPEED_PULSE_PER_LOOP)
            right_pulses = np.clip(right_pulses, -self.MAX_SPEED_PULSE_PER_LOOP,
                                 self.MAX_SPEED_PULSE_PER_LOOP)
            
            # Send command
            command = f"m {left_pulses} {right_pulses}\n"
            response = self.send_command(command)
            
            if response != 'OK':
                self.get_logger().warn('Failed to send velocity command')
                
        except Exception as e:
            self.get_logger().error(f'Error in cmd_vel callback: {e}')

    def update_odometry(self):
        """Update robot odometry from encoder readings."""
        try:
            # Read encoders
            readings = self.read_encoders()
            if not readings:
                return
                
            left_m, right_m = readings
            
            # Calculate changes
            d_left = left_m - self.last_encoder_readings['left']
            d_right = right_m - self.last_encoder_readings['right']
            
            # Update stored readings
            self.last_encoder_readings['left'] = left_m
            self.last_encoder_readings['right'] = right_m
            
            # Calculate robot movement
            d_center = (d_right + d_left) / 2.0
            d_theta = (d_right - d_left) / self.WHEEL_SEPARATION
            
            # Update wheel positions for joint states
            self.wheel_positions['left'] += d_left / self.WHEEL_RADIUS
            self.wheel_positions['right'] += d_right / self.WHEEL_RADIUS
            
            # Get time difference
            current_time = self.get_clock().now()
            dt = (current_time - self.last_odom_time).nanoseconds / 1e9
            
            if dt > 0:
                # Update velocities
                self.twist['linear'] = d_center / dt
                self.twist['angular'] = d_theta / dt
                
                # Update pose
                self.pose['theta'] = (self.pose['theta'] + d_theta) % (2 * math.pi)
                self.pose['x'] += d_center * math.cos(self.pose['theta'])
                self.pose['y'] += d_center * math.sin(self.pose['theta'])
                
                # Publish joint states
                joint_state = JointState()
                joint_state.header.stamp = current_time.to_msg()
                joint_state.name = ['left_wheel_joint', 'right_wheel_joint']
                joint_state.position = [self.wheel_positions['left'],
                                     self.wheel_positions['right']]
                joint_state.velocity = [d_left / (dt * self.WHEEL_RADIUS),
                                     d_right / (dt * self.WHEEL_RADIUS)]
                self.joint_pub.publish(joint_state)
                
                # Create and publish odometry message
                odom = Odometry()
                odom.header.stamp = current_time.to_msg()
                odom.header.frame_id = 'odom'
                odom.child_frame_id = 'base_link'
                
                # Set pose
                odom.pose.pose.position.x = self.pose['x']
                odom.pose.pose.position.y = self.pose['y']
                odom.pose.pose.position.z = 0.0
                
                # Convert theta to quaternion
                odom.pose.pose.orientation.z = math.sin(self.pose['theta'] / 2.0)
                odom.pose.pose.orientation.w = math.cos(self.pose['theta'] / 2.0)
                
                # Set twist
                odom.twist.twist.linear.x = self.twist['linear']
                odom.twist.twist.angular.z = self.twist['angular']
                
                self.odom_pub.publish(odom)
                
                # Broadcast transform
                transform = TransformStamped()
                transform.header = odom.header
                transform.child_frame_id = odom.child_frame_id
                transform.transform.translation.x = odom.pose.pose.position.x
                transform.transform.translation.y = odom.pose.pose.position.y
                transform.transform.translation.z = odom.pose.pose.position.z
                transform.transform.rotation = odom.pose.pose.orientation
                
                self.tf_broadcaster.sendTransform(transform)
                
            self.last_odom_time = current_time
            
        except Exception as e:
            self.get_logger().error(f'Error updating odometry: {e}')

    def watchdog_timer(self):
        """Monitor command timeout and stop robot if necessary."""
        try:
            current_time = self.get_clock().now()
            if (current_time - self.last_cmd_time).nanoseconds / 1e9 > self.CMD_TIMEOUT:
                self.send_command('s\n')
                
        except Exception as e:
            self.get_logger().error(f'Error in watchdog: {e}')

    def cleanup(self):
        """Clean up resources before shutdown."""
        try:
            self.send_command('s\n')  # Stop the robot
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
    finally:
        driver.cleanup()
        driver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
