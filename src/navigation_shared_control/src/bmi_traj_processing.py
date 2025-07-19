from load_pkl_file import LoadLastNTrials
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from sklearn.model_selection import train_test_split
import pickle

def make_continuous_copy(alpha):
    alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
    continuous_x = np.zeros_like(alpha)
    continuous_x[0] = alpha[0]
    for i in range(1, len(alpha)):
        if not (np.sign(alpha[i]) == np.sign(alpha[i - 1])) and np.abs(alpha[i]) > np.pi / 2:
            continuous_x[i] = continuous_x[i - 1] + (
                    alpha[i] - alpha[i - 1]) - np.sign(
                (alpha[i] - alpha[i - 1])) * 2 * np.pi
        else:
            continuous_x[i] = continuous_x[i - 1] + (alpha[i] - alpha[i - 1])

    return continuous_x

def derivative_of(x, dt=1, radian=False):
    if radian:
        x = make_continuous_copy(x)

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
def load_bmidata_cartesian(path, target_frequecy, min_length, test_size=0.3, viz=False, aug=False):
    # trainData = {}
    trainData = []
    testData = []
    with open(path,'rb') as f:
        data = pickle.load(f)
    ee_log = data["data"]
    frequency = data["frequency"]
    dt = 1./target_frequecy
    scale = 1
    # dt = 1./frequency

    stride = int(frequency//target_frequecy)
    train_set, test_set = train_test_split(ee_log, test_size=test_size, random_state=42)
    if aug == False:
        tr_num = 1
    else:
        tr_num = 5
    # training dataset
    for l in range(len(train_set)):
        cur_sequence = (np.array(train_set[l])*scale)
        for s in range(stride):
            idx_list = np.array(range(s, cur_sequence.shape[0], stride))
            # term = cur_sequence[:,:3]
            term_ori = cur_sequence[idx_list,:2]
            if term_ori.shape[0] < min_length:
                term_ori = cur_sequence

            # for tr in range(0,tr_num):
            #     term = transform(term_ori, tr)
            # term = term_ori
            # vel_term = derivatives_of(term, dt=dt)
            # acc_term = derivatives_of(vel_term, dt=dt)
            # term = np.concatenate((term,vel_term,acc_term),axis=-1)
            # # term = term[idx_list,:]

            if term.shape[0] < min_length:
                continue
            # else:
            #     term = term[:length,:]
            trainData.append(term)

            # for tr in range(0,tr_num):
            #     term = transform_bmi(term_ori, tr)
            #     vel_term = derivatives_of(term, dt=dt)
            #     acc_term = derivatives_of(vel_term, dt=dt)
            #     term = np.concatenate((term,vel_term,acc_term),axis=-1)
            #     # term = term[idx_list,:]

            #     if term.shape[0] < min_length:
            #         continue
            #     # else:
            #     #     term = term[:length,:]
            #     trainData.append(term)
    # trainData = np.stack(trainData)

    for l in range(len(test_set)):
        cur_sequence = (np.array(test_set[l])*scale)
        idx_list = np.array(range(0, cur_sequence.shape[0], stride))
        # term = cur_sequence[:,:3]
        term = cur_sequence[idx_list,:2]
        vel_term = derivatives_of(term, dt=dt)
        acc_term = derivatives_of(vel_term, dt=dt)
        term = np.concatenate((term,vel_term,acc_term),axis=-1)
        if viz == True:
            testData.append(term)
        else:
            for j in range(term.shape[0]//min_length):
                term_j = term[j*min_length:(j+1)*min_length]
                if term_j.shape[0] < min_length:
                    continue

                testData.append(term_j)
        # if term.shape[0] < min_length:
        #     continue

        # testData.append(term)
    return trainData, testData, target_frequecy


def load_bmi_data2(neural_file_list, target_frequency, min_length, test_size=0.3):
    trials, taskParamerters, _ = LoadLastNTrials(neural_file_list, 10000,answerNumbers=[1])
    scale = 1
    trainData = []
    stride = (1000//50) // target_frequency
    dt = 1/target_frequency
    min_length = 20

    train_set, test_set = train_test_split(trials, test_size=test_size, random_state=42)

    for i in range(len(train_set)):
        traj = train_set[i].avatarTrajectory
        traj = np.stack((traj['z'],traj['x']),axis=1)
        for k in range(traj.shape[0]):
            if traj[k,0]>0:
                break
        traj = traj[k:,:]
        cur_sequence = traj*scale
        for s in range(stride):
            idx_list = np.array(range(s, cur_sequence.shape[0], stride))
            # term = cur_sequence[:,:3]
            term = cur_sequence[idx_list,:3]
            if term.shape[0] < min_length:
                continue
            vel_term = derivatives_of(term, dt=dt)
            acc_term = derivatives_of(vel_term, dt=dt)
            term = np.concatenate((term,vel_term,acc_term),axis=-1)
            trainData.append(term)
    
    testData = []
    for l in range(len(test_set)):
        traj = test_set[l].avatarTrajectory
        traj = np.stack((traj['z'],traj['x']),axis=1)
        for k in range(traj.shape[0]):
            if traj[k,0]>0:
                break
        traj = traj[k:,:]
        cur_sequence = traj*scale
        idx_list = np.array(range(0, cur_sequence.shape[0], stride))
        # term = cur_sequence[:,:3]
        term = cur_sequence[idx_list,:3]
        vel_term = derivatives_of(term, dt=dt)
        acc_term = derivatives_of(vel_term, dt=dt)
        term = np.concatenate((term,vel_term,acc_term),axis=-1)
        # for j in range(term.shape[0]//min_length):
        #     term_j = term[j*min_length:(j+1)*min_length]
        #     if term_j.shape[0] < min_length:
        #         continue

        testData.append(term)
    return trainData, testData, target_frequency

def load_bmi_data(neural_file_list, target_frequency, min_length, test_size=0.3):
    trials, taskParamerters, _ = LoadLastNTrials(neural_file_list, 10000,answerNumbers=[1])
    scale = 1
    trainData = []
    stride = (1000//50) // target_frequency
    dt = 1/target_frequency
    min_length = 20

    train_set, test_set = train_test_split(trials, test_size=test_size, random_state=42)

    for i in range(len(train_set)):
        traj = train_set[i].avatarTrajectory
        traj = np.stack((traj['z'],traj['x']),axis=1)
        for k in range(traj.shape[0]):
            if traj[k,0]>0:
                break
        traj = traj[k:,:]
        cur_sequence = traj*scale
        for s in range(stride):
            idx_list = np.array(range(s, cur_sequence.shape[0], stride))
            # term = cur_sequence[:,:3]
            term = cur_sequence[idx_list,:3]
            if term.shape[0] < min_length:
                continue
            vel_term = derivatives_of(term, dt=dt)
            acc_term = derivatives_of(vel_term, dt=dt)
            term = np.concatenate((term,vel_term,acc_term),axis=-1)
            trainData.append(term)
    
    testData = []
    for l in range(len(test_set)):
        traj = test_set[l].avatarTrajectory
        traj = np.stack((traj['z'],traj['x']),axis=1)
        for k in range(traj.shape[0]):
            if traj[k,0]>0:
                break
        traj = traj[k:,:]
        cur_sequence = traj*scale
        idx_list = np.array(range(0, cur_sequence.shape[0], stride))
        # term = cur_sequence[:,:3]
        term = cur_sequence[idx_list,:3]
        vel_term = derivatives_of(term, dt=dt)
        acc_term = derivatives_of(vel_term, dt=dt)
        term = np.concatenate((term,vel_term,acc_term),axis=-1)
        for j in range(term.shape[0]//min_length):
            term_j = term[j*min_length:(j+1)*min_length]
            if term_j.shape[0] < min_length:
                continue

            testData.append(term_j)
    return trainData, testData, target_frequency

def load_bmi_data_velo(neural_file_list, target_frequency, min_length, test_size=0.3):
    trials, taskParamerters, _ = LoadLastNTrials(neural_file_list, 10000, answerNumbers=[1])
    scale = 1
    positions = []
    stride = (1000//50) // target_frequency
    dt = 1/target_frequency
    min_length = 20
    velos = []


    for i in range(len(trials)):
        traj_pv = trials[i].avatarTrajectory
        traj = np.stack((traj_pv['z'],traj_pv['x']),axis=1)
        traj_velo = np.stack((traj_pv['vz'],traj_pv['vx']),axis=1)
        for k in range(traj.shape[0]):
            if traj[k,0]>0 and traj[k,1]>0:
                break
        traj = traj[k:,:]
        traj_velo = traj_velo[k:,:]

        positions.append(traj)
        velos.append(traj_velo)
    
    return positions, velos, target_frequency


if __name__=="__main__":
    file_list = os.listdir('/home/pinhao/Desktop/franka_sim_ws/src/navigation_shared_control/src/neural_data')
    file_list_full = []
    for file_name in file_list:
        # if file_name != exclude_file:
        file_list_full.append(os.path.join('/home/pinhao/Desktop/franka_sim_ws/src/navigation_shared_control/src/neural_data', file_name))    # Open pkl file
    positions, velos, target_frequency = load_bmi_data_velo(file_list_full, 20, 20)
    print()