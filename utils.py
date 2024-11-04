import numpy as np
import torch


def disc_l2_loss(disc_value):
    
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def adv_disc_l2_loss(real_disc_value, fake_disc_value):
    
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb

import numpy as np
import torch
import math

def rotate_Y(input, beta):
	'''
	beta: angle, range 0~360.
	'''
	output = torch.zeros_like(input)
	beta = beta * (math.pi / 180) # angle to radian
	output[..., 0] = np.cos(beta)*input[..., 0] + np.sin(beta)*input[..., 2]
	output[..., 2] = -np.sin(beta)*input[..., 0] + np.cos(beta)*input[..., 2]
	output[..., 1] = input[..., 1]

	output[..., 3] = np.cos(beta)*input[..., 3] + np.sin(beta)*input[..., 5]
	output[..., 5] = -np.sin(beta)*input[..., 3] + np.cos(beta)*input[..., 5]
	output[..., 4] = input[..., 4]
	return output

def get_adj(N, J, edges):
	adj = np.eye(N*J)
	for edge in edges:
		for i in range(N):
			adj[edge[0] + i * J, edge[1] + i * J] = 1
			adj[edge[1] + i * J, edge[0] + i * J] = 1
	return torch.from_numpy(adj).float().cuda()

def get_connect(N, J):
	conn = np.zeros((N*J, N*J))
	conn[:J, :J] = 1
	conn[J:, J:] = 1
	return torch.from_numpy(conn).float().cuda()

def distance_loss(target, pred):
    mse_loss = (pred - target) ** 2
    mse_loss = mse_loss.sum(-1)
    mse_loss = mse_loss.sqrt()
    loss = mse_loss.mean()
    return loss

def relation_loss(target, pred):
    mse_loss = torch.abs(pred - target)
    loss = mse_loss.mean()
    return loss

def process_pred(pred_vel):
	pred_vel_x = pred_vel[:, :, :50]
	pred_vel_y = pred_vel[:, :, 50:]


	return pred_vel_x, pred_vel_y