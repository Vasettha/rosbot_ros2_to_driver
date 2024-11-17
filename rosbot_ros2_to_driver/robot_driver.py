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

    # ... rest of the methods (update_odometry, publish_joint_states) remain the same ...

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
