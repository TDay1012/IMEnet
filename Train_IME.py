import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch.optim as optim
import numpy as np
import torch_dct as dct
import torch
import time
import math
import random
from IME.Models import Transformer
from utils import rotate_Y, distance_loss
from metrics import FDE, JPE, APE
from tqdm import tqdm


def batch_denormalization(data, para):
    '''s
    :data: [B, T, N, J, 6] or [B, T, J, 3]
    :para: [B, 3]
    '''
    if data.shape[2]==2:
        data[..., :3] += para[:, None, None, None, :]
    else:
        data += para[:, None, None, :]
    return data

# train dataset
from pre_data import Data
train_device = 'cuda'
dataset = Data(dataset='mocap_umpm', mode=0, device=train_device, transform=False)
batch_size = 32
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# test dataset
test_device = 'cuda'
test_dataset = Data(dataset='mocap_umpm', mode=1, device=test_device, transform=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=30, shuffle=False)

model =  Transformer(d_word_vec=128, d_model=128, d_inner=1024,n_layers=3, n_head=8, d_k=64, d_v=64, device=test_device).to(test_device)


lrate=0.001
print(">>> training params: {:.2f}M".format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))
# predictor P
optimizer = optim.AdamW(model.parameters(), lr=lrate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

# save model name
T_Model = f'IME_{batch_size}_{lrate}'
B_Model = f'IME_{batch_size}_{lrate}'

# log
log_dir = str(os.path.join(os.getcwd(), 'IME_logs/'))
model_dir = T_Model + '/'
if not os.path.exists(log_dir + model_dir):
    os.makedirs(log_dir + model_dir)

loss_list = []
error_list = []

best_eval = 100
start_time = time.time()
best_epoch = 0
losses = []  # 初始化损失值列表
# 加载训练模型
#save_path = './train_model/IME_32_0.001/epoch_99.model'
#model.load_state_dict(torch.load(save_path, map_location=train_device))

# training
for epoch in range(100):
    rc = 1
    tf = 1
    steps = 0
    test_loss_list = []
    total_loss=0
    print(f'-------------------------Epoch:{epoch}-----------------------------')
    print('Time since start:{:.1f} minutes.'.format((time.time() - start_time) / 60.0))

    lrate = optimizer.param_groups[0]['lr']
    print('Training Processing:')
    all_mpjpe = np.zeros(5)
    count = 0


    for j, train_data in tqdm(enumerate(train_dataloader)):

        input_total, _ = train_data  # 获取数据批次   B,T,N,J,D
        input_total = input_total.float().cuda()  # 将数据移动到GPU，并转换为浮点类型

        input_total[..., [1, 2]] = input_total[..., [2, 1]]  # 调整坐标轴顺序    B,T,N,J,D
        input_total[..., [4, 5]] = input_total[..., [5, 4]]  # 调整坐标轴顺序

        batch_size = input_total.shape[0]  # 获取批次大小
        n_person = input_total.shape[2]   #获取人数
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],[7, 8], [7, 9],
             [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]])


        # 如果启用了相对坐标，则计算并减去相机速度
        if rc:
            camera_vel = input_total[:, 1:75, :, :, 3:].mean(dim=(1, 2, 3))  # B, 3
            input_total[..., 3:] -= camera_vel[:, None, None, None]  # 减去相机速度
            input_total[..., :3] = input_total[:, 0:1, :, :, :3] + input_total[..., 3:].cumsum(dim=1)  # 计算相对位置

        input_total = input_total.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, 75, 6)  # 调整数据维度  B, NxJ, T, 6


        if tf:
            angle = random.random() * 360  # 生成随机旋转角度
            # 随机旋转数据
            input_total = rotate_Y(input_total, angle)
            input_total *= (random.random() * 0.4 + 0.8)  # 随机缩放数据

        input_total = input_total.view(batch_size,n_person ,15,-1,6)
        input_total = input_total.view(batch_size*n_person, 15, -1, 6)
        input_total = input_total.permute(0, 2, 1, 3)
        #BN,75,15,6

        # 获取预测特征
        input_joint = input_total[:, :50]  # BN,T, J,6

        # 位置信息
        input_seq = input_joint[:, :, :, :3].view(batch_size, n_person, 50, 15, 3)
        input_seq = input_seq.reshape(batch_size, n_person, 50, 15 * 3)  # B,N,T,JD

        # 速度信息
        input_vel = input_joint[:, :, :, 3:]  # BN,T, J,3
        input_vel = input_vel.reshape(batch_size * n_person, 50, 15 * 3)  # BN,T,JD

        # 执行模型的前向传播
        input_joint = dct.dct(input_joint)
        pred_vel = model.forward(input_vel, dct.idct(input_vel[:, -1:]), input_seq)  # BN,T,JD
        pred_vel = dct.idct(pred_vel)
        #BN,T,JD

        gt_vel = input_total[..., 3:]  # 获取真实速度
        gt_vel = gt_vel[:, 50:].reshape(batch_size * n_person, 25, 15 * 3)

        # 计算损失
        loss_pred = distance_loss(pred_vel, gt_vel)
        loss = loss_pred
        count += batch_size  # 更新样本计数器
        losses.append([loss.item()])  # 记录当前损失

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    scheduler.step()

    with open(os.path.join(log_dir + model_dir, 'log.txt'), 'a+') as log:
            log.write('Epoch: {} \n'.format(epoch))
            log.write('Lrate: {} \n'.format(lrate))
            log.write('Time since start:{:.1f} minutes.\n'.format((time.time() - start_time) / 60.0))
    save_path = os.path.join('train_model',f'{T_Model}', f'epoch_{epoch}.model')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

# Test
    # 加载模型
    model.load_state_dict(torch.load(save_path, map_location=test_device))

    frame_idx = [5, 10, 15, 20, 25]
    n = 0
    ape_err_total = np.arange(len(frame_idx), dtype=np.float_)
    jpe_err_total = np.arange(len(frame_idx), dtype=np.float_)
    fde_err_total = np.arange(len(frame_idx), dtype=np.float_)
    test_loss = 0
    loss_list1 = []
    loss_list2 = []
    loss_list3 = []
    all_mpjpe = np.zeros(5)
    count = 0
    with torch.no_grad():
        model.eval()
        print('Validating Processing:')
        for _, test_data in tqdm(enumerate(test_dataloader,0)):
            n = n+1
            input_total_original, para = test_data
            input_total_original = input_total_original.float().cuda()
            input_total = input_total_original.clone()

            batch_size = input_total.shape[0]  # 获取批次大小
            n_person = input_total.shape[2]  # 获取人数

            T = 75
            input_total[..., [1, 2]] = input_total[..., [2, 1]]
            input_total[..., [4, 5]] = input_total[..., [5, 4]]

            if rc:
                camera_vel = input_total[:, 1:75, :, :, 3:].mean(dim=(1, 2, 3))  # B, 3
                input_total[..., 3:] -= camera_vel[:, None, None, None]  # 减去相机速度
                input_total[..., :3] = input_total[:, 0:1, :, :, :3] + input_total[..., 3:].cumsum(dim=1)  # 计算相对位置

            input_total = input_total.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, 75, 6)
            # B, NxJ, T, 6

            input_total = input_total.view(batch_size, n_person, 15, -1, 6)
            input_total = input_total.view(batch_size * n_person, 15, -1, 6)
            input_total = input_total.permute(0, 2, 1, 3)
            # BN,75,15,6

            # 获取预测特征
            input_joint = input_total[:, :50]  # BN,T, J,6

            # 位置信息
            input_seq = input_joint[:, :, :, :3].view(batch_size, n_person, 50, 15, 3)
            input_seq = input_seq.reshape(batch_size, n_person, 50, 15 * 3)    # B,N,T,JD

            # 速度信息
            input_vel = input_joint[:, :, :, 3:]  # BN,T, J,3
            input_vel = input_vel.reshape(batch_size * n_person, 50, 15 * 3)    # BN,T,JD

            # 执行模型的前向传播
            input_joint = dct.dct(input_joint)
            pred_vel = model.forward(input_vel, dct.idct(input_vel[:, -1:]), input_seq)  #BN,T,JD
            pred_vel = dct.idct(pred_vel)
            pred_vel = pred_vel.view(batch_size, n_person, 25, 15, 3)


            pred_vel = pred_vel.permute(0, 2, 1, 3, 4)  #B,T,N,J,D
            pred_vel = pred_vel.reshape(batch_size, 25, -1, 3)
            if rc:
                pred_vel = pred_vel + camera_vel[:, None, None]

            pred_vel[..., [1, 2]] = pred_vel[..., [2, 1]]

            motion_gt = input_total_original[..., :3].view(batch_size, T, -1, 3)
            motion_pred = (pred_vel.cumsum(dim=1) + motion_gt[:, 49:50])

            motion_pred = batch_denormalization(motion_pred.cpu(), para)  #B,T,NJ,D
            motion_gt = batch_denormalization(motion_gt.cpu(), para)      #B,T,NJ,D

            motion_gt = motion_gt[:, 49:74, :].view(batch_size, 25, n_person, 15, 3).permute(0, 2, 1, 3, 4)
            motion_pred = motion_pred.view(batch_size, 25, n_person, 15, 3).permute(0, 2, 1, 3, 4)

            prediction = motion_gt
            gt = motion_pred
            # ->test
            ape_err = APE(gt, prediction, frame_idx)
            jpe_err = JPE(gt, prediction, frame_idx)
            fde_err = FDE(gt, prediction, frame_idx)

            ape_err_total += ape_err
            jpe_err_total += jpe_err
            fde_err_total += fde_err

        print("{0: <16} | {1:6d} | {2:6d} | {3:6d} | {4:6d} | {5:6d}".format("Lengths", 200, 400, 600, 800, 1000))
        print("=== JPE Test Error ===")
        print(
            "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}".format("Our", jpe_err_total[0] / n,
                                                                                     jpe_err_total[1] / n,
                                                                                     jpe_err_total[2] / n,
                                                                                     jpe_err_total[3] / n,
                                                                                     jpe_err_total[4] / n))
        print("=== APE Test Error ===")
        print(
            "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}".format("Our", ape_err_total[0] / n,
                                                                                     ape_err_total[1] / n,
                                                                                     ape_err_total[2] / n,
                                                                                     ape_err_total[3] / n,
                                                                                     ape_err_total[4] / n))
        print("=== FDE Test Error ===")
        print(
            "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}".format("Our", fde_err_total[0] / n,
                                                                                     fde_err_total[1] / n,
                                                                                     fde_err_total[2] / n,
                                                                                     fde_err_total[3] / n,
                                                                                     fde_err_total[4] / n))

        # ->log
        with open(os.path.join(log_dir + model_dir, 'log.txt'), 'a+') as log:
            log.write(
                "{0: <16} | {1:6d} | {2:6d} | {3:6d} | {4:6d} | {5:6d}\n".format("Lengths", 200, 400, 600, 800, 1000))
            log.write("=== APE Test Error ===\n")
            log.write(
                "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}\n".format("Our", jpe_err_total[0] / n,
                                                                                           jpe_err_total[1] / n,
                                                                                           jpe_err_total[2] / n,
                                                                                           jpe_err_total[3] / n,
                                                                                           jpe_err_total[4] / n))
            log.write("=== APE Test Error ===\n")
            log.write(
                "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}\n".format("Our", ape_err_total[0] / n,
                                                                                           ape_err_total[1] / n,
                                                                                           ape_err_total[2] / n,
                                                                                           ape_err_total[3] / n,
                                                                                           ape_err_total[4] / n))
            log.write("=== APE Test Error ===\n")
            log.write("{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}\n\n".format("Our",
                                                                                                   fde_err_total[0] / n,
                                                                                                   fde_err_total[1] / n,
                                                                                                   fde_err_total[2] / n,
                                                                                                   fde_err_total[3] / n,
                                                                                                   fde_err_total[4] / n))