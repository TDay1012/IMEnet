import torch.utils.data as data
import torch
import numpy as np
import copy
#import open3d as o3d


def normalize(data):
    '''
    Notice: without batch operation.
    '''
    mean_pose = np.mean(data[0, 0, :], axis=0)
    # shape: 3
    data = data - mean_pose[None, None, None, :]
    return data, mean_pose

def batch_denormalization(data, para):
    '''
    :data: [B, T, N, J, 6] or [B, T, J, 3]
    :para: [B, 3]
    '''
    if data.shape[2]==2:
        data[..., :3] += para[:, None, None, None, :]
    else:
        data += para[:, None, None, :]
    return data

class Data(data.Dataset):
    def __init__(self, dataset, mode=0, device='cuda', transform=False):
        if dataset == "mocap_umpm":
            self.agentsNum = 3
            if mode == 0:
                self.oridata = np.load('data/mocap/train_3_75_mocap_umpm.npy')

            else:
                self.oridata = np.load('data/mocap/test_3_75_mocap_umpm.npy')
        if dataset == "mupots":  # two modes both for evaluation
            if mode == 0:
                self.oridata = np.load(
                    'data/mupots3d/mupots_150_2persons.npy')[:, :, 1::2, :]
                self.agentsNum = 2
            if mode == 1:
                self.oridata = np.load('data/mupots3d/mupots_150_3persons.npy')[:, :, 1::2, :]
                self.agentsNum = 3
        # if dataset == "3dpw":
        #     if mode == 1:
        #         self.data = np.load(
        #             '/home/ericpeng/DeepLearning/Projects/MotionPrediction/MRT_nips2021/pose3dpw/test_2_3dpw.npy')
        if dataset == "mix1":
            if mode == 1:
                self.oridata = np.load('data/mix/mix1_6persons.npy')
                self.agentsNum = 6
        if dataset == "mix2":
            if mode == 1:
                self.oridata = np.load('data/mix/mix2_10persons.npy')
                self.agentsNum = 10

        self.data = []
        self.data_para = []

        videoNumIn = len(self.oridata)
        agentsNum = self.agentsNum
        timeStepsNum = 75
        jointsNum = 15
        coordsNum = 3  # x y z
        self.dim = 6

        for i in range(videoNumIn):

            temp_data = np.array(self.oridata[i])
            curr_data = temp_data.reshape((agentsNum, timeStepsNum, jointsNum, coordsNum))  # [N, t, J, 3]
            curr_data, curr_data_para = normalize(curr_data)
            vel_data = np.zeros((agentsNum, timeStepsNum, jointsNum, coordsNum))
            vel_data[:,1:,:,:] = (np.roll(curr_data, -1, axis=1) - curr_data)[:,:-1,:,:]
            data = np.concatenate((curr_data, vel_data), axis=3)
            self.data.append(data)
            self.data_para.append(curr_data_para)

    def __getitem__(self, idx: int):
        data = self.data[idx].transpose((1, 0, 2, 3))  # [t, N, J, 3]
        para = self.data_para[idx]
        return data, para

    def __len__(self):
        return len(self.data)



