#!/usr/bin/env python
#coding=utf-8

import sys
import os
# Add workspace root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from model.trajectron import Trajectron
from queue import Queue
from threading import Thread
import time

import zmq
import zmqmsg
import logging
import tns
import sys

from mppi import MobileMPPI
import torch.distributions as td

from predictor_assistance.PredictorAssistance import PredictorAssistance2
from predictor_assistance.msg import Pose, PoseArray

def derivative_of(x, dt=1):
    not_nan_mask = ~np.isnan(x)
    masked_x = x[not_nan_mask]

    if masked_x.shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=(masked_x[1] - masked_x[0])) / dt
    # dx[not_nan_mask] = np.ediff1d(masked_x, to_begin=0.0) / dt

    return dx

def derivatives_of(x, dt=1, radian=False):
    timestep, dim = x.shape
    dxs = []
    for d in range(dim):
        dxs.append(derivative_of(x[:,d],dt))
    dxs = np.stack(dxs,axis=-1)
    return dxs


def map2d_bilinear_generation2(goal_list, obs_list, scale, resolution, radius=1):
    # assert resolution%2 == 1
    N = resolution
    dim = 2
    radius = radius
    map_tensor = np.zeros((3, resolution, resolution), dtype=np.float32)
    xx, yy = np.meshgrid(np.linspace(0, resolution-1, resolution), np.linspace(0, resolution-1, resolution), indexing='ij')

    map_coord = np.stack((xx,yy), axis=-1) - np.ones((1,1,2)) * (resolution//2) # (N, N, dim)

    center_x = center_y = resolution//2
    map_tensor[0, center_x, center_y] = 1.0

    map_coord = map_coord.reshape((-1, dim))

    if len(obs_list) != 0:
        obs = np.stack(obs_list, axis=0) * scale # (N_goals, dim)

        obs_rel_dist = map_coord[:, None] - obs[None] # (N*N, N_goals, 2)
        obs_rel_dist[obs_rel_dist>=radius] = 0
        obs_rel_dist[obs_rel_dist<=-radius] = 0
        obs_rel_dist = np.abs(obs_rel_dist)
        obs_rel_dist[obs_rel_dist>0] = radius - obs_rel_dist[obs_rel_dist>0] 

        map_tensor[2,:,:] = (obs_rel_dist[:,:,0].reshape(N,N,-1) * obs_rel_dist[:,:,1].reshape(N,N,-1)).sum(-1)

    # goal mapping
    if len(goal_list) != 0:
        goals = np.stack(goal_list, axis=0) * scale # (N_goals, dim)

        goal_rel_dist = map_coord[:, None] - goals[None] # (N*N, N_goals, 2)
        goal_rel_dist[goal_rel_dist>=radius] = 0
        goal_rel_dist[goal_rel_dist<=-radius] = 0
        goal_rel_dist = np.abs(goal_rel_dist)
        goal_rel_dist[goal_rel_dist>0] = radius - goal_rel_dist[goal_rel_dist>0]

        map_tensor[1,:,:] = (goal_rel_dist[:,:,0].reshape(N,N,-1) * goal_rel_dist[:,:,1].reshape(N,N,-1)).sum(-1)

    
    map_tensor = np.clip(map_tensor, 0, 1)

    # img = cv2.resize(map_tensor.transpose(1,2,0), (224,224))*255.0
    # cv2.imshow("img",img.astype(np.uint8))
    # cv2.waitKey(1)

    # img = cv2.resize(map_tensor.transpose(1,2,0), (224,224))
    # plt.imshow(img)
    # plt.show()


    # plt.show()
    return map_tensor




class timer:
    def __init__(self, timelimit=0.1):
        self.start = None
        self.timelimit = timelimit
        self.lag = 0
    
    def set_start(self):
        self.start = time.time()

    # def is_timesup(self):
    #     lag = (time.time() - self.start) - self.timelimit
    #     while lag < 0:
    #         lag = (time.time() - self.start) - self.timelimit
    #     self.lag = lag
    #     print(self.lag)
    #     self.start = None
    #     return 
    
    def is_timesup(self):
        while (time.time() - self.start) - self.timelimit < 0:
            time.sleep(0.004)

        self.start = None
        return 
    
    def refresh(self,):
        self.lag = 0.0
        self.start = None
    

class trajectron_service:
    def __init__(self, trajectron, ph, logger, method_idx=1, modes=10, timelimit=0.1):
        self.trajectron = trajectron
        self.method_idx = method_idx
        self.logger = logger

        # hyper-parameter for shared control
        self.gamma = 2 # amplification coefficient for GAF
        self.nu = 0.95 # mamxium guidance of GAF
        self.kappa = 30 # kappa for von mises distribution p_r_u 
        self.probability_balance_coeff = 10


        # variable for trajectron prediction
        self.dist = None        
        self.traj_log = Queue()
        self.op_count = 0
        self.interval = 1
        self.modes = modes
        self.ph = ph
        self.dt = 1/trajectron.hyperparams['frequency']
        self.ph_limit = 100
        self.scale = 1
        self.dim = 2

        # zmq communication
        self.context = zmq.Context()
        # self.publisher = zmqmsg.Publisher(self.context, port=33456)
        # self.subscriber = zmqmsg.Subscriber(self.context, port=33458)

        self.publisher = self.context.socket(zmq.REQ)
        self.subscriber = self.context.socket(zmq.REP)
        # self.publisher.connect(tns.zmq.Address("localhost", 33456))
        # self.subscriber.connect(tns.zmq.Address("localhost", 33458))

        # remoteAddress = "localhost"
        self.publisher.connect(tns.zmq.Address("localhost", 33457))
        self.subscriber.connect(tns.zmq.Address("localhost", 33459))

        # workspace init
        self.workspace_limits = np.asarray([[0, 10.], [-9, 9], [-10, 10]])
        self.x_range = self.workspace_limits[0]
        self.y_range = self.workspace_limits[1]
        self.past_count = 0

        x_min = self.x_range[0]
        x_max = self.x_range[1]
        y_min = self.y_range[0]
        y_max = self.y_range[1]
        self.resolution = 1.5
        self.analysis_resolution = 0.5
        self.XY_ana = torch.meshgrid([torch.arange(x_min, x_max, self.analysis_resolution), torch.arange(y_min, y_max, self.analysis_resolution)],indexing='ij')
        self.XY = torch.meshgrid([torch.arange(x_min, x_max, self.resolution), torch.arange(y_min, y_max, self.resolution)],indexing='ij')
        self.prob_grid = None
        # self.grid_msg = OccupancyGrid()
        # self.grid_msg.info.resolution = self.resolution
        # self.grid_msg.info.width = torch.arange(x_min, x_max, self.resolution).shape[0]
        # self.grid_msg.info.height = torch.arange(y_min, y_max, self.resolution).shape[0]
        # self.grid_msg.info.origin.position.x = 0.0
        # self.grid_msg.info.origin.position.y = -9.0
        # self.grid_msg.info.origin.position.z = 0.0
        # self.grid_msg.info.origin.orientation.x = 0.0
        # self.grid_msg.info.origin.orientation.y = 0.0
        # self.grid_msg.info.origin.orientation.z = 0.0
        # self.grid_msg.info.origin.orientation.w = 1.0
        # self.grid_msg.data = [0 for i in range(self.grid_msg.info.width*self.grid_msg.info.height)]
        self.device = self.trajectron.model.device

        self.mpc = MobileMPPI(horizon=self.ph, dim=2, gamma=0.9, device=self.device, XY=self.XY, dt=self.dt)

        #### user model ##########
        self.user_sigma = 0.05
        self.user_model = td.MultivariateNormal(torch.ones(1, self.dim).to(self.device)*(-0.1), torch.eye(self.dim).to(self.device)*self.user_sigma)
        self.goal_position = 3* np.ones((2,2))
        self.obs_position = 2* np.ones((2,2))
        self.take_control_threshold = 1.5
        self.safety_count = 0



        ########### HO+APF ###############
        if self.method_idx == 3:
            self.HO = PredictorAssistance2()


        self.timer = timer(timelimit)

        return
    
        

    def trajectory_prediction(self, pos_msg):
        """
        pos: array(3,)
        """
        t1 = None
        pred_v = [0,0,0]
        try:
            current_pos = np.array([pos_msg.position.x, pos_msg.position.y])
        except:
            current_pos = np.array([pos_msg[0], pos_msg[1]])
        if len(self.traj_log.queue) > 0:
            # current_pos = np.array(pos[:2])
            last_pos = self.traj_log.queue[-1]
            if np.sqrt(((current_pos-last_pos)**2).sum()) > 4:
                self.traj_log = Queue()
                self.dist = None
                self.trajectron.model.entropy_ub = None
                self.trajectron.model.entropy_lb = None
        self.traj_log.put(current_pos[:2])
        self.op_count += 1
        if len(self.traj_log.queue) > 10:
            self.traj_log.get()
        # validate whether out of workspace
        # if current_pos[0] < self.workspace_limits[0,0] or current_pos[0] > self.workspace_limits[0,1] or current_pos[1] < self.workspace_limits[1,0] or current_pos[1] > self.workspace_limits[1,1]:
        #     return
        if len(self.traj_log.queue) >= 4 and self.op_count % self.interval == 0:
            t1 = time.time()
            trajectory = self.nparray2Traj(np.array(list(self.traj_log.queue)), goals=self.goal_position, obstacles=self.obs_position)
            with torch.no_grad():
                y_dist, v_dist, predictions = self.trajectron.predict(trajectory,
                                            self.ph,
                                            num_samples=1,
                                            z_mode=False,
                                            gmm_mode=False,
                                            all_z_sep=False,
                                            full_dist=True,
                                            dist=True,
                                            measure=self.user_model)

            self.dist = y_dist
            self.v_dist = v_dist

            pred_v = self.v_dist.get_at_time(0).mode().detach().cpu().numpy().reshape(-1)
            pred_v = pred_v.tolist()


        return pred_v
    
    def nparray2Traj(self, pose_array, goals=None, obstacles=None):
        # num_steps = should be 10
        num_steps = pose_array.shape[0]
        data = pose_array
        term = data*self.scale
        vel_term = derivatives_of(term, dt=self.dt)
        acc_term = derivatives_of(vel_term, dt=self.dt)
        data = np.concatenate((term,vel_term,acc_term),axis=-1)
        first_history_index = torch.LongTensor(np.array([0])).cuda()
        x = data[2:,:]
        y = np.zeros((12,6))
        std = np.array([3,3,2,2,1,1])
        # std = np.array([1,1,1,1,1,1,1,1,1])
        dim = std.shape[-1]

        rel_state = np.zeros_like(x[0])
        rel_state[0:dim//3] = np.array(x)[-1, 0:dim//3]

        if goals is not None:
            goals = np.array(goals)[:, 0:dim//3]
            goals = (goals - rel_state[0:dim//3])#/std[:dim//3]
        if obstacles is not None:
            obstacles = np.array(obstacles)[:, 0:dim//3]
            obstacles = (obstacles - rel_state[0:dim//3])#/std[:dim//3]

        # map_tensor = map2d_bilinear_generation(goals.tolist(), obstacles.tolist(), 3, 25)
        map_tensor = map2d_bilinear_generation2(goals.tolist(), obstacles.tolist(), 1, 25, radius=1.8)
        map_tensor = torch.tensor(map_tensor, dtype=torch.float)

        x_st = np.where(np.isnan(x), np.array(np.nan), (x - rel_state) / std)
        y_st = np.where(np.isnan(y), np.array(np.nan), y / std)
        x_t = torch.tensor(x, dtype=torch.float).unsqueeze(0).cuda()
        y_t = torch.tensor(y, dtype=torch.float).unsqueeze(0).cuda()
        x_st_t = torch.tensor(x_st, dtype=torch.float).unsqueeze(0).cuda()
        y_st_t = torch.tensor(y_st, dtype=torch.float).unsqueeze(0).cuda()
        context = {}
        if goals is not None:
            goals = torch.tensor(goals, dtype=torch.float).cuda()
        else:
            goals = torch.zeros((0,2), dtype=torch.float).cuda()
        if obstacles is not None:
            obstacles = torch.tensor(obstacles, dtype=torch.float).cuda()
        else:
            obstacles = torch.zeros((0,2), dtype=torch.float).cuda()
        context = {
            "goals": [goals],
            "obstacles": [obstacles],
        }

        # batch = (first_history_index, x_t, y_t[...,2:4], x_st_t, y_st_t[...,2:4], context)
        batch = (first_history_index, x_t, y_t[...,dim//3:2*dim//3], x_st_t, y_st_t[...,dim//3:2*dim//3], {'map':map_tensor.unsqueeze(0).cuda()})
        return batch
    

    def nparray2PoseArray(self, traj_array):
        traj_array = traj_array.reshape(-1,3)
        traj_posearray = PoseArray()
        ph = traj_array.shape[0]
        for i in range(ph):
            term = Pose()
            term.position.x = traj_array[i,0]
            term.position.y = traj_array[i,1]
            term.position.z = traj_array[i,2]
            traj_posearray.poses.append(term)
        return traj_posearray
    

    def mpc_control(self, goal_position, obs_position, cur_position, velocity):
        
        v_r, mpc_traj = self.mpc.plan_action(cur_position[:self.dim], velocity , self.dist, self.v_dist, goal_position, obs_position)
        # velocity[:self.dim]

        return v_r.tolist(), mpc_traj
    
    
    def poseArray2nparray(self, pose_array_msg):
        # num_steps = should be 10
        num_steps = len(pose_array_msg.poses)
        data = []
        for i in range(num_steps):
            data.append(np.array([pose_array_msg.poses[i].position.x, pose_array_msg.poses[i].position.y, pose_array_msg.poses[i].position.z]))
        data = np.stack(data,axis=0)
        return data
    
    def receiver(self):
        self.logger.info("receiver started")
        trial_start_t = None
        last_velo = None
        while True:
            # Note this is going to wait until it sees a new experiment, which also means that if
            # an experiment is currently running this will wait for the next experiment.
            try:
                identifier, experiment = zmqmsg.ReceiveMessage(self.subscriber, timeout=100)
            except OSError:
                continue
            if identifier == "exit":
                self.logger.info("Exit")
                break
            elif identifier == "ping":
                zmqmsg.SendMessage(self.publisher, "pong")
                continue
            elif identifier != "Experiment":
                continue
            # This is the time to create a new logfile with this name.
            self.logger.info(f"New experiment:{experiment['task']}")
            # if experiment['task'] == "movingCamera":
            #     self.interval = 1
            # else:
            #     self.interval = 2
            # zmqmsg.SendMessage(self.publisher, "Alive")
            while True:
                identifier, message = zmqmsg.ReceiveMessage(self.subscriber)
                if identifier == "end":
                    if trial_start_t is not None:
                        time_spend = time.time() - trial_start_t
                        self.logger.info("time_spend:{:.4f}".format(time_spend))
                    self.logger.info("Experiment finished")
                    break
                elif identifier == "StartTrial":
                    if trial_start_t is not None:
                        time_spend = time.time() - trial_start_t
                        self.logger.info("time_spend:{:.4f}".format(time_spend))
                    self.logger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    self.logger.info(f"Trial start:{message['index']}")
                    self.past_count = 0
                    self.safety_count = 0
                    last_velo = None
                    self.timer.refresh()
                elif identifier == "ShowTarget":
                    # This one has the target position.
                    last_velo = None
                    goal_position = np.array(message["targetPosition"]).reshape(-1,3) # (M*3,)->(M,3)
                    self.goal_position = np.stack((goal_position[:,-1], goal_position[:,0]), axis=1)
                    self.logger.info('Goal List')
                    for g_idx in range(goal_position.shape[0]):
                        self.logger.info("Goal{}:{}".format(g_idx, goal_position[g_idx].tolist()))
                    self.logger.info('End Goal List')
                    try:
                        obs_position = np.array(message["obstaclePosition"]).reshape(-1,3)
                        self.obs_position = np.stack((obs_position[:,-1], obs_position[:,0]), axis=1)
                    except:
                        obs_position = np.zeros((0,3))
                        self.obs_position = np.zeros((0,2))
                    self.logger.info('Obstacle List')
                    for o_idx in range(obs_position.shape[0]):
                        self.logger.info("Obstacle{}:{}".format(o_idx, obs_position[o_idx].tolist()))
                    if self.method_idx == 3:
                        self.HO.InitPred(self.goal_position, np.array([0,0,0,0,0,0,1]))
                        obs_position = np.concatenate((self.obs_position, np.zeros((self.obs_position.shape[0], 1))),axis=-1)
                        goal_position = np.concatenate((self.goal_position, np.zeros((self.goal_position.shape[0],1))),axis=-1)
                        self.HO.create_potential(None, obs_position=obs_position, goal_pos_list=goal_position)
                    self.logger.info('End Obstacle List')

                elif identifier == "stop":
                    self.logger.info("Trial stop")
                elif identifier == "AvatarInfo":
                    if self.past_count == 0:
                        trial_start_t = time.time()
                    
                    newposition = message["avatarPosition"]
                    cur_position = [newposition['z'], newposition['x'], newposition['y']] # real exp

                    if last_velo is not None:
                        cur_position = np.array(cur_position) + last_velo*self.dt
                    newVelocity = message["avatarVelocity"] # bug xyz -> zxy
                    self.logger.info("Velocity input:{}".format([newVelocity[0],newVelocity[1],newVelocity[2]]))
                    newVelocity = [message["avatarVelocity"][-1], message["avatarVelocity"][0], 0]
                    if (np.array(newVelocity)**2).sum() < 0.01:
                        zmqmsg.SendMessage(self.publisher, "Velocity", tuple([0,0,0]))
                        continue
                    self.logger.info("================= iteration:{}".format(self.past_count))
                    self.logger.info("avatar position:{}".format([newposition['x'], newposition['y'], newposition['z']]))
                    self.past_count+=1
                    self.safety_count-=1

                    pose_msg = Pose()
                    pose_msg.position.x = cur_position[0]
                    pose_msg.position.y = cur_position[1]
                    pose_msg.position.z = cur_position[2]
                    pose_msg.orientation.x = 0.0
                    pose_msg.orientation.y = 0.0
                    pose_msg.orientation.z = 0.0
                    pose_msg.orientation.w = 1.0
                    
                    self.timer.set_start()
                    t1 = time.time()
                    if self.method_idx == 1:
                        final_velo = np.array(message["avatarVelocity"]) 
                        velo_norm = np.linalg.norm(final_velo)
                        if velo_norm > 3:
                            final_velo = (final_velo*3/velo_norm).tolist()
                        
                        self.timer.is_timesup()
                        # print("latency:", time.time()-t1)

                    if self.method_idx == 2:
                        # self._eepose_pub.publish(pose_msg)
                        # self.user_model = None
                        # t1 = time.time()
                        obs_distance = np.linalg.norm((np.array(cur_position[:2])+self.dt*np.array(newVelocity[:2]))[None]-self.obs_position,axis=1) # (N)
                        if (self.obs_position.shape[0] > 0 and obs_distance.min() < self.take_control_threshold):
                            self.safety_count = 3
                        if self.safety_count >0:
                            self.user_model = None
                            print("AI take control!!!!")
                        else:
                            self.user_model = td.MultivariateNormal(torch.tensor(np.array(newVelocity[:2]).reshape(-1,self.dim), dtype=torch.float32).to(self.device), torch.eye(self.dim).to(self.device)*self.user_sigma)
                        self.trajectory_prediction(pose_msg)
                        if self.trajectron.model.entropy_ub is not None:
                            self.logger.info("entropy:{}".format(self.trajectron.model.entropy_ub))
                    
                        if self.dist is not None and (np.linalg.norm(newVelocity) > 0.01):
                            # prob, predicted_velo = self.score(self.goal_position, cur_position, newVelocity) # z-x -> xyz
                            predicted_velo, mpc_traj = self.mpc_control(self.goal_position, self.obs_position, cur_position, np.array(newVelocity[:2])) # z-x -> xyz
                            exe_traj = mpc_traj.pop()
                            exe_traj = np.concatenate((exe_traj, np.zeros(exe_traj[...,-1:].shape)),axis=-1)
                            self.traj_msg = self.nparray2PoseArray(exe_traj)
                            multi_mode_traj = []
                            for i in range(len(mpc_traj)):
                                term = np.concatenate((mpc_traj[i], np.zeros(mpc_traj[i][...,-1:].shape)),axis=-1)
                                multi_mode_traj.append(self.nparray2PoseArray(term/self.scale))
                            self.traj_msg_list = multi_mode_traj
                            final_velo = [predicted_velo[1], 0, predicted_velo[0]]
                        else:
                            final_velo = message["avatarVelocity"]
                        velo_norm = np.linalg.norm(np.array(final_velo))
                        if velo_norm > 3:
                            final_velo = np.array(final_velo) 
                            final_velo = (final_velo*3/velo_norm).tolist()
                        
                        self.timer.is_timesup()
                        # print("latency:", time.time()-t1)

                    if self.method_idx == 3:
                        
                        ################ HO ##############
                        predicted_velo = self.HO.get_CAtwist(np.array(newVelocity[:3]+[0,0,0,]), pose_msg)
                        final_velo = [predicted_velo[1], 0, predicted_velo[0]]
                        
                        final_velo = np.array(final_velo) 
                        velo_norm = np.linalg.norm(final_velo)
                        if velo_norm > 3:
                            final_velo = (final_velo*3/velo_norm).tolist()
                        
                        self.timer.is_timesup()
                        # print("latency:", time.time()-t1)

                    
                    self.logger.info(f"Velocity back:{final_velo}")
                    zmqmsg.SendMessage(self.publisher, "Velocity", tuple(final_velo))
                    last_velo = np.array([final_velo[2], final_velo[0], final_velo[1]])




def init_service():
    import json
    methods_dict = {1:'direct',
                    2:'RTV2',
                    3:'HO'}

    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)

    # method_idx = 3
    method_idx = 2
    timelimit = 0.1


    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        if torch.cuda.device_count() == 1:
            # If you have CUDA_VISIBLE_DEVICES set, which you should,
            # then this will prevent leftover flag arguments from
            # messing with the device allocation.
            device = 'cuda:0'

        device = torch.device(device)


    # This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
    torch.cuda.set_device(device)


    # Load hyperparameters from json
    config_path = "config_bmi.json"
    if not os.path.exists(config_path):
        print('Config json not found!')
    with open(config_path, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['edge_encoding'] = False
    hyperparams['batch_size'] = 128
    hyperparams['k_eval'] = 10
    hyperparams['map_encoding'] = True

    
    hyperparams["frequency"] = 10


    log_writer = None
    model_dir = None
    zmqOnly = False
    model = torch.load("model_checkpoint.pth")


    trajectron = Trajectron(hyperparams,
                            log_writer,
                            device)


    trajectron.model.node_modules = model
    trajectron.set_annealing_params()

    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    trajectron.model.to(device)
    trajectron.model.eval()

    traj_service = trajectron_service(trajectron, ph=12, method_idx=method_idx, logger=logger, timelimit=timelimit)
    return traj_service
if __name__=='__main__':
    methods_dict = {1:'direct',
                    2:'RTV2',
                    3:'HO'}

    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)

    from datetime import datetime

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    user_name = 'test'
    os.makedirs("logs/{}".format(user_name), exist_ok=True)
    

    method_idx = 2
    timelimit = 0.1

    method = methods_dict[method_idx]
    log_file_path = os.path.expanduser(os.path.join("logs/{}".format(user_name), '_'.join((current_time,user_name,method))+".log"))
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        if torch.cuda.device_count() == 1:
            # If you have CUDA_VISIBLE_DEVICES set, which you should,
            # then this will prevent leftover flag arguments from
            # messing with the device allocation.
            device = 'cuda:0'

        device = torch.device(device)


    # This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
    torch.cuda.set_device(device)


    # Load hyperparameters from json
    config_path = "trajectron/src/config_bmi.json"
    if not os.path.exists(config_path):
        print('Config json not found!')
    with open(config_path, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['edge_encoding'] = False
    hyperparams['batch_size'] = 128
    hyperparams['k_eval'] = 10
    hyperparams['map_encoding'] = True

    
    hyperparams["frequency"] = 10


    log_writer = None
    model_dir = None
    zmqOnly = False

    trajectron = Trajectron(hyperparams,
                            log_writer,
                            device)

    model = torch.load("/home/u0161364/Robot-TrajectronV2/checkpoints/Exp42_maxent_autoalpha_93.pth")

    trajectron.model.node_modules = model
    trajectron.set_annealing_params()

    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    trajectron.model.to(device)
    trajectron.model.eval()

    traj_service = trajectron_service(trajectron, ph=12, method_idx=method_idx, logger=logger, timelimit=timelimit)

    if zmqOnly:
        traj_service.receiver()
        sys.exit(0)

    print("trajectron loaded successfully")
    traj_service.receiver()
    # t0 = Thread(target=traj_service.receiver, name='zmq_receier')
    # t0.start()


    

        