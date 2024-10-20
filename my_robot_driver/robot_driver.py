import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from example_interfaces.srv import Trigger
import serial
import math

class RobotDriver(Node):
    def __init__(self):
        super().__init__('robot_driver')
        self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        
        # Robot-specific constants
        self.PULSES_PER_METER = 2343
        self.PID_LOOPS_PER_SECOND = 40
        self.MAX_SPEED_PULSE_PER_LOOP = 25
        self.WHEEL_SEPARATION = 0.23  # Adjust as needed
        
        # Create a subscriber for velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10)
        
        # Create a service server for odometry
        self.odom_service = self.create_service(
            Trigger,
            'get_odometry',
            self.get_odometry_callback)

        # Variables to store robot's position
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.prev_left_distance = 0.0
        self.prev_right_distance = 0.0

    def cmd_vel_callback(self, msg):
        # Convert m/s to pulse per pid loop
        linear_speed = msg.linear.x * self.PULSES_PER_METER / self.PID_LOOPS_PER_SECOND
        angular_speed = msg.angular.z * self.WHEEL_SEPARATION / 2 * self.PULSES_PER_METER / self.PID_LOOPS_PER_SECOND

        # Calculate left and right wheel speeds
        left_speed = int(linear_speed - angular_speed)
        right_speed = int(linear_speed + angular_speed)
        
        # Clamp speeds to the allowed range
        left_speed = max(-self.MAX_SPEED_PULSE_PER_LOOP, min(self.MAX_SPEED_PULSE_PER_LOOP, left_speed))
        right_speed = max(-self.MAX_SPEED_PULSE_PER_LOOP, min(self.MAX_SPEED_PULSE_PER_LOOP, right_speed))
        
        # Send command to robot
        command = f"m {left_speed} {right_speed}\n"
        self.serial_port.write(command.encode())
        self.get_logger().info(f"Sent command: {command.strip()}")

    def get_odometry_callback(self, request, response):
        try:
            # Request encoder counts
            self.serial_port.write(b"e\n")
            serial_response = self.serial_port.readline().decode().strip()
            
            # Parse the response
            left_str, right_str = serial_response.split("Right encoder (mm):")
            left_distance = float(left_str.split("Left encoder (mm):")[1].strip())
            right_distance = float(right_str.strip())

            # Convert mm to meters
            left_distance_m = left_distance / 1000
            right_distance_m = right_distance / 1000

            # Calculate change in distance
            delta_left = left_distance_m - self.prev_left_distance
            delta_right = right_distance_m - self.prev_right_distance

            # Update previous distances
            self.prev_left_distance = left_distance_m
            self.prev_right_distance = right_distance_m

            # Calculate robot's movement
            delta_distance = (delta_right + delta_left) / 2
            delta_theta = (delta_right - delta_left) / self.WHEEL_SEPARATION

            # Update robot's position and orientation
            self.x += delta_distance * math.cos(self.theta + delta_theta / 2)
            self.y += delta_distance * math.sin(self.theta + delta_theta / 2)
            self.theta += delta_theta

            # Prepare response
            response.success = True
            response.message = f"x: {self.x:.3f}, y: {self.y:.3f}, theta: {self.theta:.3f}"
            return response

        except Exception as e:
            self.get_logger().error(f"Error getting odometry data: {e}")
            response.success = False
            response.message = f"Error: {str(e)}"
            return response

def main(args=None):
    rclpy.init(args=args)
    node = RobotDriver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()