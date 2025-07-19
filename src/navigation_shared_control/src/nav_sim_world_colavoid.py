import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import os
import glob
import random
import rospy
from navigation_shared_control.srv import cartMove,VelMove,Execute
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from navigation_shared_control.msg import KeyCommand
from nav_msgs.msg import Path
import tf2_ros
from threading import Thread, Lock
from geometry_msgs.msg import Twist, TransformStamped, Transform
import zmq
import zmqmsg
import tns

def keyboard_detection(panda ,velo):
    pub_command = rospy.Publisher("/user_command", KeyCommand, queue_size=1)
    twist_msg = Twist()
    command_msg = KeyCommand()
    while True:
        command_msg.header.stamp = rospy.Time.now()

        #Create zero twist message
        twist_msg.linear.x = 0
        twist_msg.linear.y = 0
        twist_msg.linear.z = 0
        twist_msg.angular.x = 0
        twist_msg.angular.y = 0
        twist_msg.angular.z = 0
        g = p.getKeyboardEvents()

        if ord('x') in g:
            command_msg.command = 5
        elif 32 in g:
            command_msg.command = 6
        else:
            command_msg.command = command_msg.TWIST
        # if len(g.keys()) == 0:
        #     continue
        if p.B3G_UP_ARROW in g:
            twist_msg.linear.x = velo
        
        if p.B3G_LEFT_ARROW in g:
            twist_msg.linear.y = velo
        
        if p.B3G_DOWN_ARROW in g:
            twist_msg.linear.x = -velo
        
        if p.B3G_RIGHT_ARROW in g:
            twist_msg.linear.y = -velo
        
        if ord('a') in g:
            twist_msg.linear.z = velo
        
        if ord('z') in g:
            twist_msg.linear.z = -velo

        if ord('h') in g:
            panda.ready_pose2()

        
        command_msg.twist = twist_msg
        pub_command.publish(command_msg)
        time.sleep(0.01)

class BMINavigationSim(object):
    def __init__(self, offset):
        self.offset = np.array(offset)
        self.LINK_EE_OFFSET = 0.05
        self.initial_offset = 0.05
        self.workspace_limits = np.asarray([[0, 10.], [-9, 9], [-10, 10]])
        self.scale = 1
        self._numGoals = np.random.randint(1, 6)
        self._numObstacles = np.random.randint(1, 6)
        self._urdfRoot = pd.getDataPath()
        self._blockRandom = 0.3
        self.object_root = "/home/pinhao/Desktop/franka_sim"

        self.eepose_pub = rospy.Publisher('/EE_pose', Pose, queue_size=1)
        # self.cart_move_srv = rospy.Service('/CartMove', cartMove, self.handle_move_command)
        self.vel_move_srv = rospy.Service('/VelMove', VelMove, self.handle_move_vel_command)
        # self.grasp_srv = rospy.Service('/grasp_srv', Execute, self.pickPlaceRoutine)
        self._rviz_past_pub = rospy.Publisher("/rviz_traj_past", Path, queue_size=1)
        # self._trajectory_follower = rospy.Service("/TrajMove", cartMove, self.handle_traj_move_command)
        # self.stop_srv = rospy.Service('/Stop', cartMove, self.handle_stop_command)
        
        # self.traj_pred_sub = rospy.Subscriber('/Traj_pred', PoseArray, self.traj_pred_handler)
        self.service = rospy.ServiceProxy('/VelMove', VelMove)

        self.past_trajectory = Path()
        self.future_trajectory = None

        self.context = zmq.Context()
        # self.publisher = zmqmsg.Publisher(self.context, port=33458)
        # self.subscriber = zmqmsg.Subscriber(self.context, port=33456)
        self.subscriber = self.context.socket(zmq.REP)
        self.publisher = self.context.socket(zmq.REQ)
        self.subscriber.bind(tns.zmq.Address("*", 33456))
        self.publisher.bind(tns.zmq.Address("*", 33458))
        # rospy.wait_for_service('/trajectron')
        # self.trajectron_srv = rospy.ServiceProxy('/trajectron', Trajectory)
        # self.trajectron_viz_srv = rospy.Service('/trajectron_viz', Trajectory, self.trajectron_visualizer)
        # self.object_list = self.object_setup()
        self.flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        self.init_orn=[0.0, 0.0, 0, 1.0]
        self.key_commands = [0,0,0]

        # p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])

        # p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), 0.5000000,0.00000,-.620000,0.000000,0.000000,0.0,1.0)

        x_line_id = p.addUserDebugLine([0, 0.5, 0.01],[1, 0.5, 0.01],[1,0,0])
        y_line_id = p.addUserDebugLine([1, 0.5, 0.01],[1, -0.5, 0.01],[1,0,0])
        self.plane_path = os.path.join(pd.getDataPath(), "plane.urdf")
        self.plane = p.loadURDF(
            self.plane_path,
            np.array([0, 0, -0.6]) + self.offset,
        )

        self.avatar_path = os.path.join(pd.getDataPath(), "sphere_small.urdf")
        self.avatar = p.loadURDF(self.avatar_path, np.array([0,0,0])+self.offset, self.init_orn, useFixedBase=True, flags=self.flags, globalScaling=20)
        self.goal_ids, self.obstacle_ids  = self.setting_objects(globalScaling=30)
        self.goal_ids = set(self.goal_ids)
        self.obstacle_ids = set(self.obstacle_ids)
        # self.escape_ids = set(self.escape_ids)
        self.control_dt = 0.01
        self.place_poses = [-0.00018899307178799063, -0.3069845139980316, 0.48534566164016724]
        self.z_T = 0.1
        self.reset()

        # self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.ee_pose = Pose()

        # frame_start_postition, frame_posture = p.getLinkState(self.panda,11)[4:6]
        # R_Mat = np.array(p.getMatrixFromQuaternion(frame_posture)).reshape(3,3)
        # x_axis = R_Mat[:,0]
        # x_end_p = (np.array(frame_start_postition) + np.array(x_axis*5)).tolist()
        # x_line_id = p.addUserDebugLine(frame_start_postition,x_end_p,[1,0,0])# y 轴
        # y_axis = R_Mat[:,1]
        # y_end_p = (np.array(frame_start_postition) + np.array(y_axis*5)).tolist()
        # y_line_id = p.addUserDebugLine(frame_start_postition,y_end_p,[0,1,0])# z轴
        # z_axis = R_Mat[:,2]
        # z_end_p = (np.array(frame_start_postition) + np.array(z_axis*5)).tolist()
        # z_line_id = p.addUserDebugLine(frame_start_postition,z_end_p,[0,0,1])

        return
    
    def reset(self):
        self.past_trajectory = Path()
        p.resetBasePositionAndOrientation(self.avatar, [0,0,0],[0,0,0,1])


    def setting_objects(self, globalScaling):
        goal_ids = []
        obstacle_ids = []
        banned_radius = 3
        obstacle_extension = 3
        all_objects_pos = np.zeros((0, 2))
        self.goal_pos = []
        self.obs_pos = []
        # blue cube
        # object_path = os.path.join(pd.getDataPath(), "cube_small.urdf")
        object_path = os.path.join(self.object_root, "goal_small.urdf")
        # object_path = os.path.join(pd.getDataPath(), "cube_small.urdf")
        # pos = np.array([[6,-7,0], [8.5,-3.5,0], [9.2,0,0], [8.5,3.5,0], [6,7, 0]])
        for i in range(np.random.randint(1,self._numGoals+1)):
            # if file in ['cube_5.sdf', 'cube_6.sdf']:
            #     continue
            while True:
                rand_pos = np.random.rand(2)
                rand_pos[0] = (
                    rand_pos[0]
                    * (self.workspace_limits[0, 1] - self.workspace_limits[0, 0])
                    + self.workspace_limits[0, 0]
                )
                rand_pos[1] = (
                    rand_pos[1]
                    * (self.workspace_limits[1, 1] - self.workspace_limits[1, 0])
                    + self.workspace_limits[1, 0]
                )

                if (
                    all_objects_pos.shape[0] != 0
                    and np.linalg.norm(rand_pos - all_objects_pos, axis=-1).min() < 1
                ):
                    continue

                if np.linalg.norm(rand_pos) > banned_radius+obstacle_extension:
                    break
            all_objects_pos = np.concatenate(
                (all_objects_pos, rand_pos.reshape(-1, 2)), axis=0
            )

            # rand_pos[1] = (2*rand_pos[0] - 1)*0.5
            rand_pos = np.append(rand_pos, 0.0)
            uid = p.loadURDF(
                object_path,
                rand_pos,
                self.init_orn,
                useFixedBase=True,
                flags=self.flags,
                globalScaling=globalScaling,
            )
            goal_ids.append(uid)
            translation,_ = p.getBasePositionAndOrientation(uid)
            self.goal_pos.append(translation[1])
            self.goal_pos.append(translation[2])
            self.goal_pos.append(translation[0])

        object_path = os.path.join(self.object_root, "obstacle_small.urdf")
        for i in range(np.random.randint(1,self._numObstacles+1)):
            # if file in ['cube_5.sdf', 'cube_6.sdf']:
            #     continue
            while True:
                rand_pos = np.random.rand(2)
                rand_pos[0] = (
                    rand_pos[0]
                    * (self.workspace_limits[0, 1] - self.workspace_limits[0, 0])
                    + self.workspace_limits[0, 0]
                )
                rand_pos[1] = (
                    rand_pos[1]
                    * (self.workspace_limits[1, 1] - self.workspace_limits[1, 0])
                    + self.workspace_limits[1, 0]
                )

                if (
                    all_objects_pos.shape[0] != 0
                    and np.linalg.norm(rand_pos - all_objects_pos, axis=-1).min() < 1
                ):
                    continue

                if np.linalg.norm(rand_pos) > banned_radius and np.linalg.norm(rand_pos) < banned_radius + obstacle_extension:
                    break
            all_objects_pos = np.concatenate(
                (all_objects_pos, rand_pos.reshape(-1, 2)), axis=0
            )

            # rand_pos[1] = (2*rand_pos[0] - 1)*0.5
            rand_pos = np.append(rand_pos, 0.0)
            uid = p.loadURDF(
                object_path,
                rand_pos,
                self.init_orn,
                useFixedBase=True,
                flags=self.flags,
                globalScaling=globalScaling,
            )
            obstacle_ids.append(uid)
            translation,_ = p.getBasePositionAndOrientation(uid)
            self.obs_pos.append(translation[1])
            self.obs_pos.append(translation[2])
            self.obs_pos.append(translation[0])
        return goal_ids, obstacle_ids

    def handle_move_vel_command(self, req):
        success = True
        pos_vel = req.ee_vel
        # try:
        p.resetBaseVelocity(self.avatar, linearVelocity=pos_vel[:3])
        # visualization
        current_pos = PoseStamped()
        current_pos.header.stamp = rospy.Time.now()
        current_pos.header.frame_id = 'world'
        current_pos.pose = self.ee_pose
        self.past_trajectory.poses.append(current_pos)

        # except:
        #     print("no success")
        #     success = False
        
        return success
    
    def key_command(self,twist_msg):
        
        self.key_commands = [twist_msg.twist.linear.x, twist_msg.twist.linear.y, 0]
        time.sleep(0.01)
    
        return
    
    def rviz_past_publisher(self):
        while True:
            self.past_trajectory.header.stamp = rospy.Time.now()
            self.past_trajectory.header.frame_id = 'world'
            self._rviz_past_pub.publish(self.past_trajectory)
            time.sleep(0.01)

    
    def EEPosePub(self):
        while True:
            translation,orientation = p.getBasePositionAndOrientation(self.avatar)
            pose_msg = Pose()
            pose_msg.position.x = translation[0]
            pose_msg.position.y = translation[1]
            pose_msg.position.z = translation[2]
            self.eepose_pub.publish(pose_msg)
            self.ee_pose = pose_msg
            time.sleep(0.01)
        return
    

    

    def rviz_object_publisher(self):
        while True:
            try:
                for idx in self.goal_ids:
                    translation,orientation = p.getBasePositionAndOrientation(idx)
                    static_transformStamped = TransformStamped()

                    static_transformStamped.header.stamp = rospy.Time.now()
                    static_transformStamped.header.frame_id = "world"
                    static_transformStamped.child_frame_id = "object{}".format(idx)

                    static_transformStamped.transform.translation.x = translation[0]
                    static_transformStamped.transform.translation.y = translation[1]
                    static_transformStamped.transform.translation.z = translation[2]
                    static_transformStamped.transform.rotation.x = orientation[0]
                    static_transformStamped.transform.rotation.y = orientation[1]
                    static_transformStamped.transform.rotation.z = orientation[2]
                    static_transformStamped.transform.rotation.w = orientation[3]

                    self.broadcaster.sendTransform(static_transformStamped)
                    time.sleep(0.001)
                for idx in self.obstacle_ids:
                    translation,orientation = p.getBasePositionAndOrientation(idx)
                    static_transformStamped = TransformStamped()

                    static_transformStamped.header.stamp = rospy.Time.now()
                    static_transformStamped.header.frame_id = "world"
                    static_transformStamped.child_frame_id = "onstacles{}".format(idx)

                    static_transformStamped.transform.translation.x = translation[0]
                    static_transformStamped.transform.translation.y = translation[1]
                    static_transformStamped.transform.translation.z = translation[2]
                    static_transformStamped.transform.rotation.x = orientation[0]
                    static_transformStamped.transform.rotation.y = orientation[1]
                    static_transformStamped.transform.rotation.z = orientation[2]
                    static_transformStamped.transform.rotation.w = orientation[3]

                    self.broadcaster.sendTransform(static_transformStamped)
                    time.sleep(0.001)
            except:
                pass
        return


if __name__=="__main__":
    rospy.init_node("nav_sim")
    p.connect(p.GUI)
    # p.connect(p.DIRECT)
    # p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP,1)
    
    p.setAdditionalSearchPath(pd.getDataPath())

    p.resetDebugVisualizerCamera(cameraDistance=1,cameraYaw=-90,\
                                cameraPitch=-40,cameraTargetPosition=[-5,0.0,1.5])
    timeStep=1./100.
    steps = 5
    p.setTimeStep(timeStep)
    p.setRealTimeSimulation(1)

    p.setGravity(0,0,-9.8)

    # traj_log = Queue()

    rospy.Rate(2)
    
    avatar = BMINavigationSim([0,0,0])

    t0 = Thread(target=keyboard_detection,name='keyboard_detection', args=(avatar,2))
    t0.start()

    t2 = Thread(target=avatar.EEPosePub,name='ee_pose_pubilisher')
    t2.start()

    t3 = Thread(target=avatar.rviz_object_publisher, name='object_rviz')
    t3.start()

    t4 = Thread(target=avatar.rviz_past_publisher, name='rviz_past_pubilisher')
    t4.start()

    sub_command = rospy.Subscriber("/user_command", KeyCommand, avatar.key_command)


    ##### initialization #####
    zmqmsg.SendMessage(avatar.publisher, "Experiment", {"task": "fixedCamera"})
    zmqmsg.SendMessage(avatar.publisher, "StartTrial", {"index": 0})
    zmqmsg.SendMessage(avatar.publisher, "ShowTarget", {"targetPosition": tuple(avatar.goal_pos)})
    sample = np.linspace(0,5,7)
    for i in range(len(sample)-1):
        cur_velo = [0.1,0,0.2]
        zmqmsg.SendMessage(
                avatar.publisher,
                    "AvatarInfo",
                    {
                        "avatarPosition": {
                            "z": sample[i],
                            "x": sample[i],
                            "y": 0.0,
                            "avatarRotation": 0.0,
                        },
                        "avatarVelocity": cur_velo,
                    }, timeout=None
                )
        identifer, message = zmqmsg.ReceiveMessage(avatar.subscriber, timeout=None)
        time.sleep(0.1)
    # time.sleep(0.1)
    # cur_velo = [0.1,0,0.2]
    # zmqmsg.SendMessage(
    #         avatar.publisher,
    #             "AvatarInfo",
    #             {
    #                 "avatarPosition": {
    #                     "z": sample[-1],
    #                     "x": sample[-1],
    #                     "y": 0.0,
    #                     "avatarRotation": 0.0,
    #                 },
    #                 "avatarVelocity": cur_velo,
    #             }, timeout=None
    #         )
    # identifer, message = zmqmsg.ReceiveMessage(avatar.subscriber, timeout=None)
    # time.sleep(0.1)

    zmqmsg.SendMessage(avatar.publisher, "Experiment", {"task": "fixedCamera"})
    zmqmsg.SendMessage(avatar.publisher, "StartTrial", {"index": 0})
    zmqmsg.SendMessage(avatar.publisher, "ShowTarget", {"targetPosition": tuple(avatar.goal_pos),
                                                        "obstacles": tuple(avatar.obs_pos),})
    while True:
        velo = [0,0]
        if avatar.key_commands[0]!=0.0 or avatar.key_commands[1]!=0.0:
            current_pos = avatar.ee_pose
            messagepos = [current_pos.position.x, current_pos.position.y] 
            cur_velo = [avatar.key_commands[1], 0, avatar.key_commands[0]]
            zmqmsg.SendMessage(
                    avatar.publisher,
                        "AvatarInfo",
                        {
                            "avatarPosition": {
                                "z": current_pos.position.x,
                                "x": current_pos.position.y,
                                "y": 0.0,
                                "avatarRotation": 0.0,
                            },
                            "avatarVelocity": cur_velo,
                        }, timeout=None
                    )
            print('send velo:', cur_velo)
            # time.sleep(0.1)
            # try:
            identifer, message = zmqmsg.ReceiveMessage(avatar.subscriber, timeout=None)
            if identifer == "Velocity":
                velo = message
                # velo = cur_velo
                # print(f"Got velocity back: {message}")
                # except OSError:
                #     print('error')
                #     pass
            if np.isnan(np.asarray(velo)).any() == True:
                velo = [0,0,0]
            final_velo = [velo[-1], velo[0], velo[1]]

            print(f"Got velocity back: {final_velo}")
            # except OSError:
            #     pass
            avatar.service.call(final_velo)
        else:
            # time.sleep(0.1)
            avatar.service.call([0,0,0])
        

    rospy.spin()
    print()

