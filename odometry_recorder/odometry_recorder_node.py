import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Odometry
from math import atan2, asin, pi
from os.path import expanduser

home = expanduser("~")

class OdometryRecorder(Node):

    def __init__(self):
        super().__init__('odometry_recorder')
        self.declare_parameter('topic_names', ['odometry0', 'odometry1'])
        self.declare_parameter('output_folder', home)
        self.declare_parameter('rpy_orientation', False)

        self.topic_names = self.get_parameter('topic_names').get_parameter_value().string_array_value
        output_folder = self.get_parameter('output_folder').get_parameter_value().string_value
        self.folder_path = f'{home}/ros2_iron_ws/src/odometry_recorder/data/{output_folder}'
        self.rpy_orientation = self.get_parameter('rpy_orientation').get_parameter_value().bool_value
        self.get_logger().info('Data will be saved to: ' + self.folder_path)

        self.my_callback_group = ReentrantCallbackGroup()
        self.subs = list()
        for topic in self.topic_names: 
            self.subs.append(
                self.create_subscription(
                    Odometry,
                    topic,
                    lambda msg: self.listener_callback(msg, topic),
                    10,
                    callback_group=self.my_callback_group))
            self.get_logger().info('Subscripted to topic: ' + topic)
        self.get_logger().info('Odometry recorder node initialized')

    def listener_callback(self, msg, topic):
        topic = topic.replace('/', '_')[1:]
        file = open(f'{self.folder_path}/{topic}.csv', 'a+', newline='')
        self.get_logger().info('Writing data from topic: ' + topic + ' to file: ' + file.name)
        stamp = msg.header.stamp
        position = msg.pose.pose.position
        quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        
        if (self.rpy_orientation):
            rpy = self.quat2eul(quat)
            file.write(
                str(stamp.sec) + ',' + 
                str(position.x) + ',' + 
                str(position.y) + ',' + 
                str(position.z) + ',' +
                str(rpy[0]) + ',' +
                str(rpy[1]) + ',' +
                str(rpy[2]) + '\n')
        else:
            file.write(
                str(stamp.sec) + ',' + 
                str(position.x) + ',' + 
                str(position.y) + ',' + 
                str(position.z) + ',' +
                str(quat[0]) + ',' +
                str(quat[1]) + ',' +
                str(quat[2]) + ',' +
                str(quat[3]) + '\n')
    
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
    recorder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()