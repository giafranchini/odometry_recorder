import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry

from math import atan2, asin, pi

class OdometryRecorder(Node):

    def __init__(self):
        super().__init__('odometry_recorder')
        self.declare_parameter('topic_name', 'odometry')
        self.declare_parameter('output_file', 'odometry_output.csv')
        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        self.file_path = self.get_parameter('output_file').get_parameter_value().string_value
        self.file = open(self.file_path, 'w', newline='')
        self.file.write('time,x,y,yaw\n') 
        self.subscription = self.create_subscription(
            Odometry,
            self.topic_name,
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        print('Odometry recorder node initialized')

    def listener_callback(self, msg):
        self.get_logger().info('Writing data to csv file')
        stamp = msg.header.stamp
        position = msg.pose.pose.position
        quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        rpy = self.quat2eul(quat)
        self.file.write(str(stamp.sec) + ',' + str(round(position.x, 4)) + ',' + str(round(position.y, 4)) + ',' + str(round(rpy[2], 4)) + '\n')
    
    def quat2eul(self, quat):
        discr = quat[3]*quat[1] - quat[2]*quat[0]
        if discr > 0.499999:
            roll = 0
            pitch = pi/2
            yaw = -2*atan2(quat[0], quat[3])
        if discr < -0.499999:
            roll = 0
            pitch = -pi/2
            yaw = 2*atan2(quat[0], quat[3])
        else:
            roll = atan2((2*(quat[3]*quat[0] + quat[1]*quat[2])),(1-2*(quat[0]**2 + quat[1]**2)))
            pitch = asin(2*(quat[3]*quat[1] - quat[2]*quat[0]))
            yaw = atan2((2*(quat[3]*quat[2] + quat[0]*quat[1])),(1-2*(quat[1]**2 + quat[2]**2)))
        return [roll, pitch, yaw]

def main(args=None):
    rclpy.init(args=args)

    recorder = OdometryRecorder()
    rclpy.spin(recorder)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    recorder.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()