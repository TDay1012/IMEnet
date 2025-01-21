import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch_dct as dct
from Model.short_Models import Transformer
#from model.model_mc import Transformer
from metrics import FDE, JPE, APE
import numpy as np
from data_short import Data
from tqdm import tqdm


# mocap_umpm/mupots/3dpw/mix1/mix2
test_device = 'cuda'
test_dataset = Data(dataset='mocap_umpm', mode=1, device=test_device, transform=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False)
device = 'cuda'

model = Transformer(d_word_vec=128, d_model=128, d_inner=1024, n_layers=3, n_head=8, d_k=64, d_v=64, device=device).to(device)


plot = False
gt = False
print('Data download completed')
model.load_state_dict(torch.load('./train_model/32_0.001_short/epoch_90.model', map_location=device))

body_edges = np.array(
    [[0, 1], [1, 2], [2, 3], [0, 4],
     [4, 5], [5, 6], [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]]
)

losses = []

n = 0
frame_idx = [5, 10, 15, 20, 25]
ape_err_total = np.zeros(len(frame_idx), dtype=np.float_)
jpe_err_total = np.zeros(len(frame_idx), dtype=np.float_)
fde_err_total = np.zeros(len(frame_idx), dtype=np.float_)


with torch.no_grad():
    print('Validating Processing:')
    model.eval()
    prediction_all = []
    for _, test_data in tqdm(enumerate(test_dataloader, 0)):
        input_seq, output_seq = test_data
        B, N, T, D = input_seq.shape  # 获取批次大小、人数、时间步和特征维度
        n += 1

        # ->tensor
        input_seq = torch.as_tensor(input_seq, dtype=torch.float32).to(test_device)
        output_seq = torch.as_tensor(output_seq, dtype=torch.float32).to(test_device)

        ''' first 1 second predict future 1 second '''

        input_ = input_seq.view(-1, 50, input_seq.shape[-1])
        output_ = output_seq.view(output_seq.shape[0] * output_seq.shape[1], -1, input_seq.shape[-1])
        input_ = dct.dct(input_)

        rec_ = model.forward(input_[:, 1:50, :] - input_[:, :49, :], dct.idct(input_[:, -1:, :]), input_seq)
        rec = dct.idct(rec_)
        # future 1s'results
        results = output_[:, :1, :]
        # offset to sum
        for i in range(1, 11):
            results = torch.cat([results, output_[:, :1, :] + torch.sum(rec[:, :i, :], dim=1, keepdim=True)], dim=1)
        results = results[:, 1:, :]

        ''' first 2 second predict future 1 second '''
        new_input_seq = torch.cat(
            [input_seq[:, :, 10:], results.reshape(input_seq.shape[0], input_seq.shape[1], -1, input_seq.shape[-1])],
            dim=-2)
        new_input = dct.dct(new_input_seq.reshape(-1, 50, 45))

        new_rec_ = model.forward(new_input[:, 1:, :] - new_input[:, :-1, :], dct.idct(new_input[:, -1:, :]),
                                      new_input_seq)
        new_rec = dct.idct(new_rec_)
        # future 2s'results
        new_results = new_input_seq.reshape(-1, 50, 45)[:, -1:, :]

        for i in range(1, 11):
            new_results = torch.cat([new_results,
                                     new_input_seq.reshape(-1, 50, 45)[:, -1:, :] + torch.sum(new_rec[:, :i, :], dim=1,
                                                                                              keepdim=True)], dim=1)
        new_results = new_results[:, 1:, :]

        results = torch.cat([results, new_results], dim=-2)
        rec = torch.cat([rec, new_rec], dim=-2)
        results = output_[:, :1, :]
        # offset to sum
        for i in range(1, 11 + 10):
            results = torch.cat([results, output_[:, :1, :] + torch.sum(rec[:, :i, :], dim=1, keepdim=True)], dim=1)

        results = results[:, 1:, :]

        ''' first 3 second predict future 1 second '''
        new_new_input_seq = torch.cat(
            [input_seq[:, :, 20:], results.reshape(input_seq.shape[0], input_seq.shape[1], -1, input_seq.shape[-1])],
            dim=-2)
        new_new_input = dct.dct(new_new_input_seq.reshape(-1, 50, 45))

        new_new_rec_ = model.forward(new_new_input[:, 1:, :] - new_new_input[:, :-1, :],
                                          dct.idct(new_new_input[:, -1:, :]), new_new_input_seq)

        new_new_rec = dct.idct(new_new_rec_)
        rec = torch.cat([rec, new_new_rec[:, :5]], dim=-2)

        results = output_[:, :1, :]

        for i in range(1, 21 + 5):
            results = torch.cat([results, output_[:, :1, :] + torch.sum(rec[:, :i, :], dim=1, keepdim=True)], dim=1)

        results = results[:, 1:, :]

        # 将所有预测结果拼接在一起
        prediction = results.view(B, N, -1, 15, 3)
        gt = output_seq.view(B, N, -1, 15, 3)

        # 计算误差
        ape_err = APE(gt, prediction, frame_idx)
        jpe_err = JPE(gt, prediction, frame_idx)
        fde_err = FDE(gt, prediction, frame_idx)

        ape_err_total += ape_err
        jpe_err_total += jpe_err
        fde_err_total += fde_err

        prediction_all.append(prediction)
        if prediction_all:  # 确保列表非空
            prediction_all_tensor = torch.cat(prediction_all, dim=0).cpu().numpy()
        else:
            prediction_all_tensor = np.array([])  # 或者其他合适的默认值

    # np.save('data/MuPoTs3D/IMEnet_sm_periods.npy', prediction_all_tensor)#B,N,T,J,D

        # ->print
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
