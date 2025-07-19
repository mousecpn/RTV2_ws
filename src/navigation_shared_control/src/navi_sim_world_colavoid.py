import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import os
import glob
import random
import rospy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped


from navigation_shared_control.msg import KeyCommand
from nav_msgs.msg import Path
import tf2_ros
from threading import Thread, Lock
from geometry_msgs.msg import Twist, TransformStamped, Transform
import zmq
import zmqmsg
import tns
from scipy.spatial.transform import Rotation

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
        self.workspace_limits = np.asarray([[0, 10.], [-9, 9], [-10, 10]])
        self.scale = 1
        self._numGoals = np.random.randint(2, 6)
        self._numObstacles = np.random.randint(3, 6)
        self._urdfRoot = pd.getDataPath()
        self.object_root = "objects"

        self.eepose_pub = rospy.Publisher('/EE_pose', Pose, queue_size=1)
        self._rviz_past_pub = rospy.Publisher("/rviz_traj_past", Path, queue_size=1)

        self.past_trajectory = Path()
        self.future_trajectory = None

        self.context = zmq.Context()
        
        self.subscriber = self.context.socket(zmq.REP)
        self.publisher = self.context.socket(zmq.REQ)
        # self.subscriber.bind(tns.zmq.Address("*", 33457))
        # self.publisher.bind(tns.zmq.Address("*", 33459))
        self.subscriber.bind(tns.zmq.Address("*", 33457))
        self.publisher.bind(tns.zmq.Address("*", 33459))

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
        # self.escape_ids = set(self.escape_ids)
        self.place_poses = [-0.00018899307178799063, -0.3069845139980316, 0.48534566164016724]

        self.goal_ids, self.obstacle_ids = set(), set()
        self.reset()

        # self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.ee_pose = Pose()
        return
    
    def reset(self, seed=0):
        self.past_trajectory = Path()
        p.resetBasePositionAndOrientation(self.avatar, [0,0,0],[0,0,0,1])
        if len(self.goal_ids) >0:
            self.clear_table()
        self.rng = np.random.RandomState(seed)
        self.goal_ids, self.obstacle_ids  = self.setting_objects(globalScaling=30)
        self.goal_ids = set(self.goal_ids)
        self.obstacle_ids = set(self.obstacle_ids)
        

    def is_goal_achieved(self):
        agent_pos,_ = p.getBasePositionAndOrientation(self.avatar)
        for goal_idx in self.goal_ids:
            goal_pos,_ = p.getBasePositionAndOrientation(goal_idx)
            if np.linalg.norm(np.array(agent_pos)-np.array(goal_pos), ord=2, axis=-1) < 0.5:
                return True
        return False
    
    def is_goalx_achieved(self, goal_idx):
        agent_pos,_ = p.getBasePositionAndOrientation(self.avatar)
        goal_pos,_ = p.getBasePositionAndOrientation(goal_idx)
        if np.linalg.norm(np.array(agent_pos)-np.array(goal_pos), ord=2, axis=-1) < 0.5:
            return True
        return False
    
    def is_collision(self):
        agent_pos,_ = p.getBasePositionAndOrientation(self.avatar)
        for obstacle_idx in self.obstacle_ids:
            obs_pos,_ = p.getBasePositionAndOrientation(obstacle_idx)
            if np.linalg.norm(np.array(agent_pos)-np.array(obs_pos), ord=2, axis=-1) < 1.3:
                return True
        return False
    
    
    def setting_objects(self,globalScaling):
        goal_ids = []
        obstacle_ids = []
        self.goal_pos = []
        self.obs_pos = []
        object_path = os.path.join(self.object_root, "goal_small.urdf")
        potential_pos = [[6,-7,0], [8.5,-3.5,0], [9.2,0,0], [8.5,3.5,0], [6,7, 0]]
        random.shuffle(potential_pos)
        potential_pos = np.array(potential_pos)
        for i in range(potential_pos.shape[0]):
            if i == 0:
                uid = p.loadURDF(os.path.join(self.object_root, "true_goal_small.urdf"), potential_pos[i], self.init_orn, useFixedBase=True, flags=self.flags, globalScaling=globalScaling)
                self.true_goal_idx= uid
            else:
                uid = p.loadURDF(object_path, potential_pos[i], self.init_orn, useFixedBase=True, flags=self.flags, globalScaling=globalScaling)
            goal_ids.append(uid)
            translation,_ = p.getBasePositionAndOrientation(uid)
            self.goal_pos.append(translation[1])
            self.goal_pos.append(translation[2])
            self.goal_pos.append(translation[0])
        
        z_axis = np.array([0,0,1])
        x_axis = potential_pos[0]/np.linalg.norm(potential_pos[0])
        y_axis = np.cross(z_axis, x_axis)
        
        R_mat = np.stack((x_axis,y_axis,z_axis),axis=1)
        quat = Rotation.from_matrix(R_mat).as_quat()

        object_path = os.path.join(self.object_root, "obstacle_small.urdf")
        distance_scale = np.random.rand()*0.8+1.6
        uid = p.loadURDF(
                object_path,
                potential_pos[0]/distance_scale,
                quat,
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
    
    """
    def setting_objects(self, globalScaling):
        self._numGoals = self.rng.randint(2, 4)
        self._numObstacles = self.rng.randint(3, 6)
        goal_ids = []
        obstacle_ids = []
        banned_radius = 3
        obstacle_extension = 3
        all_objects_pos = np.zeros((0, 2))
        self.goal_pos = []
        self.obs_pos = []
        # blue cube
        # object_path = os.path.join(pd.getDataPath(), "cube_small.urdf")
        
        # object_path = os.path.join(pd.getDataPath(), "cube_small.urdf")
        # pos = np.array([[6,-7,0], [8.5,-3.5,0], [9.2,0,0], [8.5,3.5,0], [6,7, 0]])

        object_path = os.path.join(self.object_root, "obstacle_small.urdf")
        for i in range(1,self._numObstacles+1):
            # if file in ['cube_5.sdf', 'cube_6.sdf']:
            #     continue
            while True:
                rand_pos = self.rng.rand(2)
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
                    and np.linalg.norm(rand_pos - all_objects_pos, axis=-1).min() < 1.
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
        obs_pos_ = np.array(self.obs_pos).reshape(-1,3)
        obs_pos_ = np.stack((obs_pos_[:,2], obs_pos_[:,0], obs_pos_[:,1]), axis=1)
        
        object_path = os.path.join(self.object_root, "goal_small.urdf")
        for i in range(1,self._numGoals+1):
            # if file in ['cube_5.sdf', 'cube_6.sdf']:
            #     continue
            while True:
                rand_pos = self.rng.rand(2)
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
                    and np.linalg.norm(rand_pos - all_objects_pos, axis=-1).min() < 1.
                ):
                    continue

                if np.linalg.norm(rand_pos) > banned_radius+obstacle_extension and isGoalOccluded(obs_pos_, rand_pos):
                    break
            all_objects_pos = np.concatenate(
                (all_objects_pos, rand_pos.reshape(-1, 2)), axis=0
            )
            
            # rand_pos[1] = (2*rand_pos[0] - 1)*0.5
            rand_pos = np.append(rand_pos, 0.0)
            if i == 1:
                uid = p.loadURDF(
                    os.path.join(self.object_root, "true_goal_small.urdf"),
                    rand_pos,
                    self.init_orn,
                    useFixedBase=True,
                    flags=self.flags,
                    globalScaling=globalScaling,
                )
                self.true_goal_idx = uid
            else:
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
        return goal_ids, obstacle_ids
    """
    

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

    def clear_table(self):
        for g in self.goal_ids:
            p.removeBody(g)
            # self.goal_ids.remove(g)
        for o in self.obstacle_ids:
            p.removeBody(o)
            # self.goal_ids.remove(o)
        return
    
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

def getInterval(a, b):
        if(a < b):
            return [a, b]
        return [b, a]


def isGoalOccluded(obstacles, goal):
    """
    Return True if there is an obstacle, False otherwise
    """
    ee_position = np.array([0,0])

    #Same dimension
    dimension_point = np.size(ee_position)
    x_range = getInterval(ee_position[0], goal[0])
    y_range = getInterval(ee_position[1], goal[1])
    
    tmp_list = list()
    for obs in obstacles:
        flag = 0
        for i in range(dimension_point):
            ax_obs = obs[i]
            if(i == 0):
                if(x_range[0] <= ax_obs <= x_range[1]):
                    flag += 1 
            elif(i == 1):
                if(y_range[0] <= ax_obs <= y_range[1]):
                    flag += 1
        if(flag == 2):
            #print("C'è un punto tra ee e goal: rimangono attivi gli escape points")
            return True
    return False

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

    rospy.Rate(20)
    
    avatar = BMINavigationSim([0,0,0])

    t0 = Thread(target=keyboard_detection,name='keyboard_detection', args=(avatar,2))
    t0.start()

    # t2 = Thread(target=avatar.EEPosePub,name='ee_pose_pubilisher')
    # t2.start()

    # t3 = Thread(target=avatar.rviz_object_publisher, name='object_rviz')
    # t3.start()

    # t4 = Thread(target=avatar.rviz_past_publisher, name='rviz_past_pubilisher')
    # t4.start()

    sub_command = rospy.Subscriber("/user_command", KeyCommand, avatar.key_command)

    epsidoes = 20
    success = 0
    collision = 0
    out_of_time = 0
    iter_limit = 200

    ##### initialization #####
    
    # zmqmsg.SendMessage(avatar.publisher, "Experiment", {"task": "movingCamera","name":"inittest", "allTargetPositions": tuple([3,3,0])}, timeout=None)
    # identifier, message = zmqmsg.ReceiveMessage(avatar.subscriber, timeout=None)
    # zmqmsg.SendMessage(avatar.publisher, "StartTrial", {"index": -1,'aiVelocityFactor':1,})
    # zmqmsg.SendMessage(avatar.publisher, "ShowTarget", {"targetPosition": tuple(avatar.goal_pos)})
    # sample = np.linspace(0,5,7)
    # for i in range(len(sample)-1):
    #     cur_velo = [0.5,0,0.2]
    #     zmqmsg.SendMessage(
    #             avatar.publisher,
    #                 "AvatarInfo",
    #                 {
    #                     "avatarPosition": {
    #                         "z": sample[i],
    #                         "x": sample[i],
    #                         "y": 0.0,
    #                         "avatarRotation": 0.0,
    #                     },
    #                     "avatarVelocity": cur_velo,
    #                 }, timeout=None
    #             )
    #     identifer, message = zmqmsg.ReceiveMessage(avatar.subscriber, timeout=None)
    #     time.sleep(0.1)
    # zmqmsg.SendMessage(avatar.publisher, "end", {})
    

    start_e = 1
    zmqmsg.SendMessage(avatar.publisher, "Experiment", {"task": "movingCamera","name":"inittest","allTargetPositions":tuple(avatar.goal_pos)})
    # identifier, message = zmqmsg.ReceiveMessage(avatar.subscriber, timeout=None)
    for e in range(start_e, start_e+epsidoes):
        iter = 0
        zmqmsg.SendMessage(avatar.publisher, "StartTrial", {"index": e,'aiVelocityFactor':1,})
        zmqmsg.SendMessage(avatar.publisher, "ShowTarget", {"targetPosition": tuple(avatar.goal_pos),
                                                            "obstaclePosition": tuple(avatar.obs_pos),})
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
                if avatar.is_collision():
                    avatar.reset(e)
                    collision += 1
                    break
                elif avatar.is_goalx_achieved(avatar.true_goal_idx):
                    avatar.reset(e)
                    success += 1
                    break
                iter += 1
                if iter > iter_limit:
                    print("out of time:",e)
                    avatar.reset(e)
                    out_of_time += 1
                    break
            else:
                # time.sleep(0.1)
                avatar.service.call([0,0,0])
    zmqmsg.SendMessage(avatar.publisher, "end", {})
    print("success rate:", success/epsidoes)
    print("collsion rate:", collision/epsidoes)     
    print("oot rate:", out_of_time/epsidoes)      
        

    rospy.spin()
    print()

