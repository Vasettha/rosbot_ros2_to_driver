import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
import serial
import math
import numpy as np
from threading import Lock
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

class RobotDriver(Node):
    def __init__(self):
        super().__init__('robot_driver')
        
        # Robot parameters
        self.WHEEL_SEPARATION = 0.23  # meters
        self.WHEEL_RADIUS = 0.0335    # meters
        self.ENCODER_TICKS_PER_METER = 2343.0
        self.MAX_LINEAR_SPEED = 0.5   # m/s
        self.MAX_ANGULAR_SPEED = 2.0  # rad/s
        self.UPDATE_RATE = 50.0       # Hz
        self.CMD_TIMEOUT = 0.5        # seconds
        
        # Initialize serial communication
        self._init_serial()
        
        # Initialize publishers, subscribers and broadcasters
        self._init_ros_interfaces()
        
        # Initialize state variables
        self._init_state()
        
        # Create timers
        self.update_timer = self.create_timer(1.0/self.UPDATE_RATE, self.update_robot_state)
        self.watchdog_timer = self.create_timer(0.1, self.watchdog)
        
        self.get_logger().info('Robot driver initialized successfully')

    def _init_serial(self):
        """Initialize serial communication with robust error handling"""
        try:
            self.serial_lock = Lock()
            self.serial_port = serial.Serial(
                port='/dev/ttyUSB1',
                baudrate=115200,
                timeout=0.02,  # Reduced timeout for faster response
                write_timeout=0.02
            )
            # Clear buffers
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to initialize serial port: {e}')
            raise

    def _init_ros_interfaces(self):
        """Initialize all ROS interfaces with appropriate QoS settings"""
        # Configure QoS profiles
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        cmd_vel_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, 'odom', odom_qos)
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, 'cmd_vel', self.cmd_vel_callback, cmd_vel_qos)
            
        # Transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

    def _init_state(self):
        """Initialize robot state variables"""
        self.pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.twist = {'linear': 0.0, 'angular': 0.0}
        self.encoder_readings = {'left': 0, 'right': 0}
        self.last_cmd_time = self.get_clock().now()
        self.last_cmd = (0, 0)

    def cmd_vel_callback(self, msg):
        """Process velocity commands with speed limiting and efficient encoding"""
        try:
            # Update command timestamp
            self.last_cmd_time = self.get_clock().now()
            
            # Apply speed limits
            linear = np.clip(msg.linear.x, -self.MAX_LINEAR_SPEED, self.MAX_LINEAR_SPEED)
            angular = np.clip(msg.angular.z, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)
            
            # Calculate wheel velocities (in m/s)
            left_speed = linear - (angular * self.WHEEL_SEPARATION / 2.0)
            right_speed = linear + (angular * self.WHEEL_SEPARATION / 2.0)
            
            # Convert to encoder ticks per control loop
            left_ticks = int(left_speed * self.ENCODER_TICKS_PER_METER / 40)  # 40Hz control loop
            right_ticks = int(right_speed * self.ENCODER_TICKS_PER_METER / 40)
            
            # Send command if different from last
            if (left_ticks, right_ticks) != self.last_cmd:
                command = f"m {left_ticks} {right_ticks}\n"
                self._send_serial_command(command)
                self.last_cmd = (left_ticks, right_ticks)
                
        except Exception as e:
            self.get_logger().error(f'Error in cmd_vel callback: {e}')

    def _send_serial_command(self, command):
        """Send command to robot with efficient error handling"""
        try:
            with self.serial_lock:
                self.serial_port.write(command.encode())
                return True
        except serial.SerialException as e:
            self.get_logger().error(f'Serial write error: {e}')
            return False

    def _read_encoders(self):
        """Read encoder values with efficient parsing"""
        try:
            with self.serial_lock:
                self.serial_port.write(b"e\n")
                response = self.serial_port.readline().decode().strip()
                
                if not response:
                    return None
                    
                # Fast string parsing without split()
                left_start = response.find(":") + 1
                right_start = response.find(":", left_start) + 1
                
                if left_start > 0 and right_start > 0:
                    try:
                        left_mm = float(response[left_start:response.find(" Right")].strip())
                        right_mm = float(response[right_start:].strip())
                        return (left_mm / 1000.0, right_mm / 1000.0)  # Convert to meters
                    except ValueError:
                        return None
                return None
                
        except serial.SerialException as e:
            self.get_logger().error(f'Serial read error: {e}')
            return None

    def update_robot_state(self):
        """Update robot state with efficient odometry calculation"""
        try:
            # Read encoders
            encoder_vals = self._read_encoders()
            if not encoder_vals:
                return
                
            # Calculate changes
            d_left = encoder_vals[0] - self.encoder_readings['left']
            d_right = encoder_vals[1] - self.encoder_readings['right']
            
            # Update stored readings
            self.encoder_readings['left'] = encoder_vals[0]
            self.encoder_readings['right'] = encoder_vals[1]
            
            # Calculate odometry
            d_center = (d_right + d_left) / 2.0
            d_theta = (d_right - d_left) / self.WHEEL_SEPARATION
            
            # Update pose
            theta = self.pose['theta'] + d_theta
            self.pose['theta'] = theta
            self.pose['x'] += d_center * math.cos(theta)
            self.pose['y'] += d_center * math.sin(theta)
            
            # Calculate velocities
            dt = 1.0 / self.UPDATE_RATE
            self.twist['linear'] = d_center / dt
            self.twist['angular'] = d_theta / dt
            
            # Publish updates
            self._publish_odometry()
            self._publish_joint_states(d_left, d_right, dt)
            
        except Exception as e:
            self.get_logger().error(f'Error updating robot state: {e}')

    def _publish_odometry(self):
        """Publish odometry with efficient message creation"""
        try:
            # Create stamped message
            now = self.get_clock().now()
            
            # Create and publish transform
            transform = TransformStamped()
            transform.header.stamp = now.to_msg()
            transform.header.frame_id = 'odom'
            transform.child_frame_id = 'base_link'
            
            # Set translation
            transform.transform.translation.x = self.pose['x']
            transform.transform.translation.y = self.pose['y']
            transform.transform.translation.z = 0.0
            
            # Calculate quaternion
            theta_2 = self.pose['theta'] / 2.0
            transform.transform.rotation.z = math.sin(theta_2)
            transform.transform.rotation.w = math.cos(theta_2)
            
            # Broadcast transform
            self.tf_broadcaster.sendTransform(transform)
            
            # Create and publish odometry
            odom = Odometry()
            odom.header = transform.header
            odom.child_frame_id = transform.child_frame_id
            
            # Copy pose and twist
            odom.pose.pose.position.x = self.pose['x']
            odom.pose.pose.position.y = self.pose['y']
            odom.pose.pose.orientation = transform.transform.rotation
            
            odom.twist.twist.linear.x = self.twist['linear']
            odom.twist.twist.angular.z = self.twist['angular']
            
            self.odom_pub.publish(odom)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing odometry: {e}')

    def _publish_joint_states(self, d_left, d_right, dt):
        """Publish joint states efficiently"""
        try:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = ['left_wheel_joint', 'right_wheel_joint']
            
            # Calculate wheel angles from distance
            angle_left = d_left / self.WHEEL_RADIUS
            angle_right = d_right / self.WHEEL_RADIUS
            
            msg.position = [angle_left, angle_right]
            msg.velocity = [angle_left/dt, angle_right/dt]
            
            self.joint_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing joint states: {e}')

    def watchdog(self):
        """Monitor command timeout"""
        try:
            if (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9 > self.CMD_TIMEOUT:
                if self.last_cmd != (0, 0):
                    self._send_serial_command("s\n")
                    self.last_cmd = (0, 0)
        except Exception as e:
            self.get_logger().error(f'Watchdog error: {e}')

    def cleanup(self):
        """Clean up resources"""
        try:
            self._send_serial_command("s\n")
            self.serial_port.close()
        except Exception as e:
            self.get_logger().error(f'Cleanup error: {e}')

def main():
    rclpy.init()
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
