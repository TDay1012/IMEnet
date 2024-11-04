import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import torch
from IME.Models import Transformer
from JRT.util import rotate_Y, get_adj, get_connect
from metrics import FDE, JPE, APE
from tqdm import tqdm
from pre_data import Data
import torch_dct as dct

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


# mocap_umpm/mupots/3dpw/mix1/mix2
test_device = 'cuda'
test_dataset = Data(dataset='mocap_umpm', mode=1, device=test_device, transform=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
device = 'cuda'

model =  Transformer(d_word_vec=128, d_model=128, d_inner=1024,n_layers=3, n_head=8, d_k=64, d_v=64, device=test_device).to(test_device)

print('Data download completed')
model.load_state_dict(torch.load('./train_model/IME_32_0.001/epoch_74.model', map_location=device))

frame_idx = [5, 10, 15, 20, 25]
n = 0
rc=1
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
    for _, test_data in tqdm(enumerate(test_dataloader, 0)):
        n = n + 1
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
        input_seq = input_seq.reshape(batch_size, n_person, 50, 15 * 3)  # B,N,T,JD

        # 速度信息
        input_vel = input_joint[:, :, :, 3:]  # BN,T, J,3
        input_vel = input_vel.reshape(batch_size * n_person, 50, 15 * 3)  # BN,T,JD

        # 执行模型的前向传播
        input_joint = dct.dct(input_joint)
        pred_vel = model.forward(input_vel, dct.idct(input_vel[:, -1:]), input_seq)  # BN,T,JD
        pred_vel = dct.idct(pred_vel)
        pred_vel = pred_vel.view(batch_size, n_person, 25, 15, 3)

        pred_vel = pred_vel.permute(0, 2, 1, 3, 4)  # B,T,N,J,D
        pred_vel = pred_vel.reshape(batch_size, 25, -1, 3)
        if rc:
            pred_vel = pred_vel + camera_vel[:, None, None]

        pred_vel[..., [1, 2]] = pred_vel[..., [2, 1]]

        motion_gt = input_total_original[..., :3].view(batch_size, T, -1, 3)
        motion_pred = (pred_vel.cumsum(dim=1) + motion_gt[:, 49:50])

        motion_pred = batch_denormalization(motion_pred.cpu(), para)  # B,T,NJ,D
        motion_gt = batch_denormalization(motion_gt.cpu(), para)  # B,T,NJ,D

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

