import pybullet as p
import pybullet_data as pd
import sys
sys.path.append("/home/pinhao/Desktop/Robot-TrajectronV2")
import math
import time
import numpy as np
import os
import glob
import random
import rospy
from navigation_shared_control.srv import cartMove,VelMove,Execute
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from navigation_shared_control.srv import Objects as ObjectsSrv
from navigation_shared_control.msg import KeyCommand
from nav_msgs.msg import Path
import tf2_ros
# from franka_sim_world import keyboard_detection
from threading import Thread, Lock
from geometry_msgs.msg import Twist, TransformStamped, Transform
import zmq
import zmqmsg
import tns
from trajectron_nav_node import trajectron_service
import torch
from model.trajectron import Trajectron
import json
import matplotlib.pyplot as plt
def  set_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


class SimpleNavigationEnv(object):
    def __init__(self, offset):
        self.offset = np.array(offset)
        self.LINK_EE_OFFSET = 0.05
        self.initial_offset = 0.05
        self.workspace_limits = np.asarray([[0, 10.], [-9, 9], [-10, 10]])
        self.scale = 1
        self._urdfRoot = pd.getDataPath()
        self._blockRandom = 0.3
        self._sdfRoot = "/home/pinhao/Desktop/franka_sim_ws/src/navigation_shared_control/models/exp1"
        self.object_root = "/home/pinhao/Desktop/franka_sim"

        self.eepose_pub = rospy.Publisher('/EE_pose', Pose, queue_size=1)
        self.vel_move_srv = rospy.Service('/VelMove', VelMove, self.handle_move_vel_command)
        self.object_srv = rospy.Service('/objects_srv', ObjectsSrv, self.handle_objects_srv)
        self._rviz_past_pub = rospy.Publisher("/rviz_traj_past", Path, queue_size=1)
        # self.stop_srv = rospy.Service('/Stop', cartMove, self.handle_stop_command)
        
        # self.traj_pred_sub = rospy.Subscriber('/Traj_pred', PoseArray, self.traj_pred_handler)
        self.service = rospy.ServiceProxy('/VelMove', VelMove)

        self.past_trajectory = Path()
        self.future_trajectory = None
        seed = 0
        self.rng = np.random.RandomState(seed)

        self.flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        self.init_orn=[0.0, 0.0, 0, 1.0]
        self.key_commands = [0,0,0]

        x_line_id = p.addUserDebugLine([0, 0.5, 0.01],[1, 0.5, 0.01],[1,0,0])
        y_line_id = p.addUserDebugLine([1, 0.5, 0.01],[1, -0.5, 0.01],[1,0,0])
        self.plane_path = os.path.join(pd.getDataPath(), "plane.urdf")
        self.plane = p.loadURDF(
            self.plane_path,
            np.array([0, 0, -0.6]) + self.offset,
        )

        self.avatar_path = os.path.join(pd.getDataPath(), "sphere_small.urdf")
        self.avatar = p.loadURDF(self.avatar_path, np.array([0,0,0])+self.offset, self.init_orn, useFixedBase=True, flags=self.flags, globalScaling=20) #
        
        self.control_dt = 0.1
        self.place_poses = [-0.00018899307178799063, -0.3069845139980316, 0.48534566164016724]
        self.z_T = 0.1

        self.goal_ids, self.obstacle_ids = set(), set()
        self.reset()

        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.ee_pose = Pose()

        return
    
    def reset(self,seed=0):
        self.rng = np.random.RandomState(seed)
        self.past_trajectory = Path()
        p.resetBasePositionAndOrientation(self.avatar, [0,0,0],[0,0,0,1])
        if len(self.goal_ids) >0:
            self.clear_table()
        self.goal_ids, self.obstacle_ids  = self.setting_objects(globalScaling=30)
        self.goal_ids = set(self.goal_ids)
        self.obstacle_ids = set(self.obstacle_ids)
        


    def setting_objects(self, globalScaling):
        self._numGoals = self.rng.randint(1, 6)
        self._numObstacles = self.rng.randint(1, 6)
        goal_ids = []
        obstacle_ids = []
        banned_radius = 3
        obstacle_extension = 3
        all_objects_pos = np.zeros((0, 2))
        self.goal_pos = []
        self.obs_pos = []
        object_path = os.path.join(self.object_root, "goal_small.urdf")
        for i in range(1,self._numGoals+1):
            while True:
                # rand_pos = np.random.rand(2)
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
        for i in range(1,self._numObstacles+1):
            while True:
                # rand_pos = np.random.rand(2)
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
                    and np.linalg.norm(rand_pos - all_objects_pos, axis=-1).min() < 1
                ):
                    continue

                if np.linalg.norm(rand_pos) > banned_radius and np.linalg.norm(rand_pos) < banned_radius + obstacle_extension:
                    break
            all_objects_pos = np.concatenate(
                (all_objects_pos, rand_pos.reshape(-1, 2)), axis=0
            )

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
    
    def clear_table(self):
        for g in self.goal_ids:
            p.removeBody(g)
            # self.goal_ids.remove(g)
        for o in self.obstacle_ids:
            p.removeBody(o)
            # self.goal_ids.remove(o)
        return
    
    def handle_objects_srv(self, flag):
        goals = PoseArray()
        for idx in self.goal_ids:
            translation,orientation = p.getBasePositionAndOrientation(idx)
            pose_msg = Pose()
            pose_msg.position.x = translation[0]
            pose_msg.position.y = translation[1]
            pose_msg.position.z = translation[2]
            pose_msg.orientation.x = orientation[0]
            pose_msg.orientation.y = orientation[1]
            pose_msg.orientation.z = orientation[2]
            pose_msg.orientation.w = orientation[3]
            goals.poses.append(pose_msg)
        
        obstacles = PoseArray()
        for idx in self.obstacle_ids:
            translation,orientation = p.getBasePositionAndOrientation(idx)
            pose_msg = Pose()
            pose_msg.position.x = translation[0]
            pose_msg.position.y = translation[1]
            pose_msg.position.z = translation[2]
            pose_msg.orientation.x = orientation[0]
            pose_msg.orientation.y = orientation[1]
            pose_msg.orientation.z = orientation[2]
            pose_msg.orientation.w = orientation[3]
            obstacles.poses.append(pose_msg)
        
        # escape_points = []
        # for idx in self.escape_ids:
        #     translation = p.getLinkState(idx,0)[0]
        #     orientation = p.getLinkState(idx,0)[1]
        #     pose_msg = Pose()
        #     pose_msg.position.x = translation[0]
        #     pose_msg.position.y = translation[1]
        #     pose_msg.position.z = translation[2]
        #     pose_msg.orientation.x = orientation[0]
        #     pose_msg.orientation.y = orientation[1]
        #     pose_msg.orientation.z = orientation[2]
        #     pose_msg.orientation.w = orientation[3]
        #     escape_points.append(pose_msg)

        return goals, obstacles
    

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
def stepSimulation(iter):
    for k in range(iter):
        p.stepSimulation()
    return

if __name__=="__main__":
    # set_random_seed(0)
    rospy.init_node("franka_sim")
    # p.connect(p.GUI)
    p.connect(p.DIRECT)
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
    
    avatar = SimpleNavigationEnv([0,0,0])

    # t0 = Thread(target=keyboard_detection,name='keyboard_detection', args=(avatar,2))
    # t0.start()
    

    t2 = Thread(target=avatar.EEPosePub,name='ee_pose_pubilisher')
    t2.start()

    t3 = Thread(target=avatar.rviz_object_publisher, name='object_rviz')
    t3.start()

    t4 = Thread(target=avatar.rviz_past_publisher, name='rviz_past_pubilisher')
    t4.start()

    # sub_command = rospy.Subscriber("/user_command", KeyCommand, avatar.key_command)
    # traj_pred = rospy.ServiceProxy('/trajectron', VeloMerge)


    from argument_parser import args

    if not torch.cuda.is_available() or args.device == 'cpu':
        args.device = torch.device('cpu')
    else:
        if torch.cuda.device_count() == 1:
            # If you have CUDA_VISIBLE_DEVICES set, which you should,
            # then this will prevent leftover flag arguments from
            # messing with the device allocation.
            args.device = 'cuda:0'

        args.device = torch.device(args.device)

    if args.eval_device is None:
        args.eval_device = torch.device('cpu')

    # This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
    torch.cuda.set_device(args.device)

    args.conf = "/home/pinhao/Desktop/Robot-TrajectronV2/config/config_bmi_test.json"
    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['edge_encoding'] = False
    hyperparams['map_encoding'] = True
    hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
    
    hyperparams["frequency"] = 10


    log_writer = None
    model_dir = None
    zmqOnly = False

    trajectron = Trajectron(hyperparams,
                            log_writer,
                            args.device)

    model = torch.load("/home/pinhao/Desktop/Robot-TrajectronV2/checkpoints/Exp42_maxent_autoalpha_93.pth")

    trajectron.model.node_modules = model
    trajectron.set_annealing_params()

    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    trajectron.model.to(args.device)
    trajectron.model.eval()

    traj_service = trajectron_service(trajectron, ph=12)
    os.makedirs('navigation_img', exist_ok=True)


    t0 = Thread(target=traj_service.rviz_multraj_publisher, name='rviz_prediction_pubilisher')
    t0.start()


    epsidoes = 100
    success = 0
    collision = 0
    out_of_time = 0
    iter_limit = 100
    for e in range(1,1+epsidoes):
        iter = 0
        # if e<=29:
        #     avatar.reset(e)
        #     continue

        ###################### viz #########################
        fig = plt.figure()
        ax = plt.axes()
        # plt.xlabel('Y-axis', fontsize=15) 
        # plt.ylabel('X-axis', fontsize=15)
        x_range = (-10, 10)
        y_range = (-1, 10)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        # plt.axis("equal")
        # ax.set_xlim([x_range[0], x_range[1]])
        # ax.set_ylim([y_range[0], y_range[1]])
        plt.tick_params(axis='both', labelsize=11)
        ax.set_aspect('equal', adjustable='datalim')
        
        width = 1.3
        # goals = goals - width/2
        # obs = obs - width/2
        for g in np.array(avatar.goal_pos).reshape(-1,3):
            # circle = plt.Rectangle((g[0], g[-1]), width=1, height=1, color='g', fill=True, linewidth=2)
            # ax.add_patch(circle)
            g = g - width/2
            circle = plt.Rectangle((g[0], g[-1]), width=width, height=width, facecolor='#4DAA59', fill=True, edgecolor='black', linewidth=1)
            ax.add_patch(circle)
        for o in np.array(avatar.obs_pos).reshape(-1,3):
            # circle = plt.Rectangle((o[0],o[-1]), width=1., height=1., color='r', fill=True, linewidth=2)
            o = o - width/2
            circle = plt.Rectangle((o[0],o[-1]), width=width, height=width, facecolor='#E76D7E', fill=True, edgecolor='black', linewidth=1)
            ax.add_patch(circle)
        trajectory = []
        ###################### viz #########################
        
        while True:
            velo = [0,0,0]
            ##### initial movement #####
            current_pos = avatar.ee_pose
            cur_pos_array = np.array([current_pos.position.y, 0, current_pos.position.x] )
            trajectory.append(np.array([current_pos.position.y, current_pos.position.x] ))
            if iter < 3:
                rand_goal_pos = np.array(avatar.goal_pos[:3])
                velo = (rand_goal_pos-cur_pos_array) / np.linalg.norm(rand_goal_pos-cur_pos_array, ord=2, axis=-1) * 0.0
                traj_service.trajectory_prediction([current_pos.position.x, current_pos.position.y])
                final_velo = [velo[-1], velo[0], velo[1]]

            # cur_velo = [avatar.key_commands[0], avatar.key_commands[1], 0]
            else:
                velo = traj_service.trajectory_prediction([current_pos.position.x, current_pos.position.y])
                if np.isnan(np.asarray(velo)).any() == True:
                    velo = [0,0,0]
                elif sum(velo) < -9999:
                    avatar.reset(e)
                    break
                else:
                    final_velo = list(velo) + [0,]
            # print(f"iter{iter}: Got velocity back: {final_velo}")
            avatar.service.call(final_velo)
            # time.sleep(0.1)
            stepSimulation(10)
            avatar.service.call([0,0,0])
            # time.sleep(0.1)

            if avatar.is_collision():
                print("col:",e)
                avatar.reset(e)
                collision += 1
                break
            elif avatar.is_goal_achieved():
                avatar.reset(e)
                success += 1
                break
            iter += 1
            if iter > iter_limit:
                print("out of time:",e)
                avatar.reset(e)
                out_of_time += 1
                break
        ###################### viz #########################
        trajectory = np.stack(trajectory, axis=0)
        ax.plot(trajectory[:,0], trajectory[ :,1], '#34638D')
        ax.scatter(trajectory[::2,0], trajectory[::2,1], s=5, c='#34638D')  
        # plt.grid(True)
        img_file_name = 'navigation_img/epsisode{}.pdf'.format(e)
        plt.savefig(img_file_name, bbox_inches='tight', pad_inches=0.1)
        ###################### viz #########################

    print("success rate:", success/epsidoes)
    print("collsion rate:", collision/epsidoes)
            
        

    # rospy.spin()
    # print()

