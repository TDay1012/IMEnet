import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch.optim as optim
import numpy as np
import torch_dct as dct
import torch
import time
from Model.short_Models import Transformer

from metrics import FDE, JPE, APE
from tqdm import tqdm



#loss
def distance_loss(target, pred):
    mse_loss = (pred - target) ** 2
    mse_loss = mse_loss.sum(-1)
    mse_loss = mse_loss.sqrt()
    loss = mse_loss.mean()
    return loss

# train dataset
from data_short import Data
train_device = 'cuda'
dataset = Data(dataset='mocap_umpm', mode=0, device=train_device, transform=False)
batch_size = 32
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
train_model = Transformer(d_word_vec=128, d_model=128, d_inner=1024, n_layers=3, n_head=8, d_k=64, d_v=64,device=train_device).to(train_device)

# test dataset
from data_short import Data
test_device = 'cuda'
test_dataset = Data(dataset='mocap_umpm', mode=1, device=test_device, transform=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=30, shuffle=False)
test_model = Transformer(d_word_vec=128, d_model=128, d_inner=1024,n_layers=3, n_head=8, d_k=64, d_v=64, device=test_device).to(test_device)

lrate=0.001
print(">>> training params: {:.2f}M".format(sum(p.numel() for p in train_model.parameters() if p.requires_grad) / 1000000.0))
# predictor P
optimizer = optim.AdamW(train_model.parameters(),lr=lrate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

# save model name
T_Model = f'ST_{batch_size}_{lrate}'
B_Model = f'ST_{batch_size}_{lrate}'

# log
log_dir = str(os.path.join(os.getcwd(), 'logs/'))
model_dir = T_Model + '/'
if not os.path.exists(log_dir + model_dir):
    os.makedirs(log_dir + model_dir)

loss_list = []
error_list = []

best_eval = 100
start_time = time.time()
best_epoch = 0

# 加载训练模型
#save_path = './train_model/32_0.001_short_2scale_channel/epoch_76.model'
#train_model.load_state_dict(torch.load(save_path, map_location=train_device))

# training
for epoch in range(150):
    total_loss = 0
    print(f'-------------------------Epoch:{epoch}-----------------------------')
    print('Time since start:{:.1f} minutes.'.format((time.time() - start_time) / 60.0))

    lrate = optimizer.param_groups[0]['lr']
    print('Training Processing:')

    for j, train_data in tqdm(enumerate(train_dataloader)):
        input_seq, output_seq = train_data
        input_seq = torch.as_tensor(input_seq, dtype=torch.float32).to(train_device)   # B,N,50,JD
        output_seq = torch.as_tensor(output_seq, dtype=torch.float32).to(train_device) # B,N,26,JD

        ''' first 50 frame predict future 10 frame '''
        input_=input_seq.view(-1, 50, input_seq.shape[-1]) # batch x n_person, 15: 15 fps -- (1 second), 45: 15joints x 3
        output_=output_seq.view(output_seq.shape[0]*output_seq.shape[1], -1, input_seq.shape[-1])
        input_ = dct.dct(input_)
        rec_=train_model.forward(input_[:, 1:50, :]-input_[:, :49, :], dct.idct(input_[:, -1:, :]), input_seq)
        rec=dct.idct(rec_) # BN,10,JD

        ''' first 2 second predict future 1 second '''
        new_input_seq=torch.cat([input_seq[:,:,10:],output_seq[:,:,1:11]],dim=-2)
        new_input_=dct.dct(new_input_seq.reshape(-1,50,45))
        new_rec_=train_model.forward(new_input_[:,1:50,:]-new_input_[:,:49,:],dct.idct(new_input_[:,-1:,:]), new_input_seq)
        new_rec=dct.idct(new_rec_) # BN,10,JD

        ''' first 3 second predict future 1 second '''
        new_new_input_seq=torch.cat([input_seq[:,:,20:],output_seq[:,:,1:21]],dim=-2)
        new_new_input_=dct.dct(new_new_input_seq.reshape(-1,50,45))
        new_new_rec_=train_model.forward(new_new_input_[:,1:,:]-new_new_input_[:,:49,:],dct.idct(new_new_input_[:,-1:,:]), new_new_input_seq)
        new_new_rec=dct.idct(new_new_rec_) # BN,10,JD

        pred_vel = torch.cat([rec, new_rec, new_new_rec[:,:5]], dim=-2)

        output_ = output_seq.view(-1, 26, output_seq.shape[-1])
        real_vel = output_[:, 1:, :] - output_[:, :-1, :]

        # Compute loss only on the predicted frames (last 45 frames)
        loss = distance_loss(pred_vel[:, :, :], real_vel[:, :, :])

        # Predictor
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.)
        optimizer.step()
    scheduler.step()


    with open(os.path.join(log_dir + model_dir, 'log.txt'), 'a+') as log:
        log.write(f'Epoch: {epoch}\n')
        log.write(f'Lrate: {lrate}\n')
        log.write(f'Time since start: {time.time() - start_time:.1f} minutes.\n')

    save_path = os.path.join('train_model', f'{T_Model}', f'epoch_{epoch}.model')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(train_model.state_dict(), save_path)

# Test
    # 加载模型
    test_model.load_state_dict(torch.load(save_path, map_location=test_device))

    n = 0
    frame_idx = [5, 10, 15, 20, 25]
    ape_err_total = np.zeros(len(frame_idx), dtype=np.float_)
    jpe_err_total = np.zeros(len(frame_idx), dtype=np.float_)
    fde_err_total = np.zeros(len(frame_idx), dtype=np.float_)
    test_loss = 0

    with torch.no_grad():
        print('Validating Processing:')
        test_model.eval()

        print('Validating Processing:')
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

            rec_ = test_model.forward(input_[:, 1:50, :] - input_[:, :49, :], dct.idct(input_[:, -1:, :]), input_seq)
            rec = dct.idct(rec_)
            # future 1s'results
            results = output_[:, :1, :]
            # offset to sum
            for i in range(1, 11):
                results = torch.cat([results, output_[:, :1, :] + torch.sum(rec[:, :i, :], dim=1, keepdim=True)], dim=1)
            results = results[:, 1:, :]

            ''' first 2 second predict future 1 second '''
            new_input_seq = torch.cat([input_seq[:,:,10:], results.reshape(input_seq.shape[0],input_seq.shape[1],-1,input_seq.shape[-1])], dim=-2)
            new_input = dct.dct(new_input_seq.reshape(-1, 50, 45))

            new_rec_ = test_model.forward(new_input[:, 1:, :] - new_input[:, :-1, :], dct.idct(new_input[:, -1:, :]), new_input_seq)
            new_rec = dct.idct(new_rec_)
            # future 2s'results
            new_results = new_input_seq.reshape(-1, 50, 45)[:, -1:, :]

            for i in range(1, 11):
                new_results = torch.cat([new_results,new_input_seq.reshape(-1, 50, 45)[:, -1:, :] + torch.sum(new_rec[:, :i, :], dim=1, keepdim=True)], dim=1)
            new_results = new_results[:, 1:, :]

            results = torch.cat([results, new_results], dim=-2)
            rec = torch.cat([rec, new_rec], dim=-2)
            results = output_[:, :1, :]
            # offset to sum
            for i in range(1, 11 + 10):
                results = torch.cat([results, output_[:, :1, :] + torch.sum(rec[:, :i, :], dim=1, keepdim=True)], dim=1)

            results = results[:, 1:, :]

            ''' first 3 second predict future 1 second '''
            new_new_input_seq = torch.cat([input_seq[:,:,20:], results.reshape(input_seq.shape[0],input_seq.shape[1],-1,input_seq.shape[-1])], dim=-2)
            new_new_input = dct.dct(new_new_input_seq.reshape(-1, 50, 45))

            new_new_rec_ = test_model.forward(new_new_input[:, 1:, :] - new_new_input[:, :-1, :],dct.idct(new_new_input[:, -1:, :]), new_new_input_seq)

            new_new_rec = dct.idct(new_new_rec_)
            rec = torch.cat([rec, new_new_rec[:,:5]], dim=-2)

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
