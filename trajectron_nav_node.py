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
from TrajectronNode import derivative_of, derivatives_of, map2d_bilinear_generation2

from mppi import MobileMPPI
import torch.distributions as td

logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

from datetime import datetime

current_time = datetime.now().strftime("%b%d_%H-%M-%S")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

# log_file_path = os.path.expanduser(os.path.join("~/logs", current_time+".log"))
# fh = logging.FileHandler(log_file_path)
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


class trajectron_service:
    def __init__(self, trajectron, ph, modes=10):
        self.trajectron = trajectron

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
        self.dt = 0.1
        self.ph_limit = 100
        self.scale = 1
        self.dim = 2

        # workspace init
        self.workspace_limits = np.asarray([[0, 10.], [-9, 9], [-10, 10]])
        self.x_range = self.workspace_limits[0]
        self.y_range = self.workspace_limits[1]
        self.past_count = 0


        self.device = self.trajectron.model.device

        self.mpc = MobileMPPI(horizon=12, dim=2, gamma=0.9, device=self.device, dt=0.1)

        #### user model ##########
        self.user_sigma = 0.5
        self.user_model = td.MultivariateNormal(torch.ones(1, self.dim).to(self.device)*(-0.1), torch.eye(self.dim).to(self.device)*self.user_sigma)
        self.goal_position = 3* np.ones((2,2))
        self.obs_position = 2* np.ones((2,2))

        self.pred_v = np.array([0,0])
        return
    
    def reset(self, goals, obstacles):
        self.traj_log = Queue()
        self.dist = None
        self.goal_position = goals
        self.obs_position = obstacles
        self.op_count = 0
        return

    def trajectory_prediction(self, pos_msg, ):
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
        self.traj_log.put(current_pos[:2])
        self.op_count += 1
        if len(self.traj_log.queue) > 10:
            self.traj_log.get()
        # validate whether out of workspace
        # if current_pos[0] < self.workspace_limits[0,0] or current_pos[0] > self.workspace_limits[0,1] or current_pos[1] < self.workspace_limits[1,0] or current_pos[1] > self.workspace_limits[1,1]:
        #     return [-9999,-9999,-9999]
        if len(self.traj_log.queue) >= 4 and self.op_count % self.interval == 0:
            t1 = time.time()
            trajectory = self.nparray2Traj(np.array(list(self.traj_log.queue)), goals=self.goal_position, obstacles=self.obs_position)
            with torch.no_grad():
                self.user_model = None
                y_dist, v_dist, predictions = self.trajectron.predict(trajectory,
                                            self.ph,
                                            num_samples=1,
                                            z_mode=False,
                                            gmm_mode=True,
                                            all_z_sep=False,
                                            full_dist=True,
                                            dist=True)
                # y_dist, v_dist, predictions = self.trajectron.predict(trajectory,
                #                             self.ph,
                #                             num_samples=1,
                #                             z_mode=True,
                #                             gmm_mode=True,
                #                             all_z_sep=False,
                #                             full_dist=False,
                #                             dist=True)
                # y_dist, v_dist, predictions = self.trajectron.predict3(trajectory,
                #                 self.workspace_limits,
                #                 num_samples=1,
                #                 z_mode=True,
                #                 gmm_mode=True,
                #                 all_z_sep=False,
                #                 full_dist=False,
                #                 dist=True,
                #                 ph_limit=100)
            # print("model inference time:", time.time()-t1)
            self.dist = y_dist
            self.v_dist = v_dist

            # pred_v = self.v_dist.get_at_time(0).mode().detach().cpu().numpy().reshape(-1)
            # pred_v = pred_v.tolist()

            # if t1 is not None:
            #     print("model processing latency:", t2-t1)
            v_r, mpc_traj = self.mpc.plan_action(current_pos[:self.dim], None, self.dist, self.v_dist, self.goal_position, self.obs_position)
            pred_v = v_r.detach().cpu().numpy().tolist()
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
        map_tensor = map2d_bilinear_generation2(goals.tolist(), obstacles.tolist(), 1, 25)
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

    
    def poseArray2Traj(self, pose_array_msg):
        # num_steps = should be 10
        num_steps = len(pose_array_msg.poses)
        data = []
        for i in range(num_steps):
            data.append(np.array([pose_array_msg.poses[i].position.x, pose_array_msg.poses[i].position.y, pose_array_msg.poses[i].position.z]))
        term = np.stack(data, axis=0)*self.scale
        vel_term = derivatives_of(term, dt=self.dt)
        acc_term = derivatives_of(vel_term, dt=self.dt)
        data = np.concatenate((term,vel_term,acc_term),axis=-1)
        first_history_index = torch.LongTensor(np.array([0])).cuda()
        x = data[2:,:]
        y = np.zeros((12,9))
        std = np.array([3,3,3,2,2,2,1,1,1])
        # std = np.array([1,1,1,1,1,1,1,1,1])

        rel_state = np.zeros_like(x[0])
        rel_state[0:3] = np.array(x)[-1, 0:3]

        x_st = np.where(np.isnan(x), np.array(np.nan), (x - rel_state) / std)
        y_st = np.where(np.isnan(y), np.array(np.nan), y / std)
        x_t = torch.tensor(x, dtype=torch.float).unsqueeze(0).cuda()
        y_t = torch.tensor(y, dtype=torch.float).unsqueeze(0).cuda()
        x_st_t = torch.tensor(x_st, dtype=torch.float).unsqueeze(0).cuda()
        y_st_t = torch.tensor(y_st, dtype=torch.float).unsqueeze(0).cuda()
        batch = (first_history_index, x_t, y_t[...,2:6], x_st_t, y_st_t[...,3:6])
        return batch

    
    
    def poseArray2nparray(self, pose_array_msg):
        # num_steps = should be 10
        num_steps = len(pose_array_msg.poses)
        data = []
        for i in range(num_steps):
            data.append(np.array([pose_array_msg.poses[i].position.x, pose_array_msg.poses[i].position.y, pose_array_msg.poses[i].position.z]))
        data = np.stack(data,axis=0)
        return data



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

    traj_service = trajectron_service(trajectron, ph=12)
    return traj_service