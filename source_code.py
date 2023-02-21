#%%
import torch
config_sys = {'task_edges_normalize':True,
              'clf_loss_weight':0.9,
              'reg_loss_weight':0.1,
              'epochs':128,
              'batch_size':64,
              'device':(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))}

#%%
import pandas as pd
import numpy as np
import tqdm

def seed_everything(seed=0):
    import os
    import numpy as np
    import torch
    import random
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

label_table_path = r''
label_table = pd.read_csv(label_table_path, header=None)

file_rootpath = r''
   
datas_info = []
for data_index in range(len(label_table.T)):
    data_info = {'subject':str(label_table.at[0, data_index]), 
                 'file_name':str(label_table.at[1, data_index]),
                 'sensor_num':int(label_table.at[2, data_index]),
                 'sensor_side':str(label_table.at[3, data_index])}
    data_info['file_path'] = file_rootpath+'\\'+data_info['subject']\
                                +'\\'+data_info['file_name']
    datas_info.append(data_info)
    del data_info

label_table = label_table.drop(index=[0,1,2,3]).reset_index(drop=True).astype(int)

tasks = []
tasks_gt = []
tasks_edges = []
progress_bar = tqdm.tqdm(range(len(datas_info)))
for data_index in progress_bar:
    progress_bar.set_description('Data Loading')
    data = pd.read_csv(datas_info[data_index]['file_path'])

    data_gt = pd.Series(data=np.nan, index=list(range(len(data))))
    for time_index in range(len(label_table)):
        
        if time_index%6==1 or time_index%6==3 or time_index%6==5:
            sp = label_table.at[time_index-1, data_index]
            ep = label_table.at[time_index, data_index]
            data_gt.loc[sp:ep] = ((time_index%6)//2)
        
        if time_index%6==5:
            sp = label_table.at[time_index-5, data_index]
            ep = label_table.at[time_index, data_index]
            task = data.loc[sp:ep, :].reset_index(drop=True)
            task_gt = data_gt.loc[sp:ep].reset_index(drop=True)
            
            tasks.append(task)
            tasks_gt.append(task_gt)
            
            index0 = label_table.at[time_index-5, data_index]
            index2 = label_table.at[time_index-3, data_index]
            index3 = label_table.at[time_index-2, data_index]
            index5 = label_table.at[time_index, data_index]
            tasks_edges.append(pd.Series([0, index2-index0, index3-index0, index5-index0]))


def zero_padding(tasks, tasks_gt, tasks_edges, padding_value=0.):
    max_len = len(max(tasks, key=len))
    padded_tasks = []
    padded_tasks_gt = []
    padded_tasks_edges = []
    padding_difs = []
    progress_bar = tqdm.tqdm(range(len(tasks)))
    for task_index in progress_bar:
        progress_bar.set_description('Zero Padding')
        dif = max_len - len(tasks[task_index])
        padding_difs.append(dif)
        if dif>0:
            padding = pd.DataFrame(padding_value, columns=list(tasks[0].columns), index=range(int(dif/2)))
            gt_padding = pd.Series(3, index=range(int(dif/2)))
            padding1 = pd.DataFrame(padding_value, columns=list(tasks[0].columns), index=range(int(dif/2)+1))
            gt_padding1 = pd.Series(3, index=range(int(dif/2)+1))
            
            padded_task = padding
            padded_task_gt = gt_padding
            padded_task = pd.concat([padded_task, tasks[task_index]], ignore_index=True)
            padded_task_gt = pd.concat([padded_task_gt, tasks_gt[task_index]], ignore_index=True)
            if dif%2 == 0:
                padded_task = pd.concat([padded_task, padding], ignore_index=True)
                padded_task_gt = pd.concat([padded_task_gt, gt_padding], ignore_index=True)
            else:
                padded_task = pd.concat([padded_task, padding1], ignore_index=True)
                padded_task_gt = pd.concat([padded_task_gt, gt_padding1], ignore_index=True)
            
            padded_tasks.append(padded_task)
            padded_tasks_gt.append(padded_task_gt)
            padded_tasks_edges.append(tasks_edges[task_index]+dif//2)
        elif dif==0:
            padded_tasks.append(tasks[task_index])
            padded_tasks_gt.append(tasks_gt[task_index])
            padded_tasks_edges.append(tasks_edges[task_index])
    return padded_tasks, padded_tasks_gt, padded_tasks_edges, padding_difs

padding_difs = []
tasks, tasks_gt, tasks_edges, padding_difs = zero_padding(tasks, tasks_gt, tasks_edges)
max_task_length = len(max(tasks, key=len))

tasks_gt_concat = pd.Series(dtype=float)
for task_gt in tasks_gt:
    tasks_gt_concat = pd.concat([tasks_gt_concat, task_gt], ignore_index=True)
print(tasks_gt_concat.value_counts(normalize=True, sort=False))

def task_edges_normalize(tasks, tasks_edges):
    for task_index in range(len(tasks)):
        task_length = len(tasks[task_index])
        tasks_edges[task_index] = tasks_edges[task_index]/task_length
    return tasks_edges
    
if config_sys['task_edges_normalize']:
    tasks_edges = task_edges_normalize(tasks, tasks_edges)

def task_edges_to_task_gt(tasks_edges, max_task_length):
    tasks_gt = []
    if config_sys['padding method']=='zero padding':
        for task_edges in tasks_edges:
            if config_sys['task_edges_normalize']:
                task_edges = (task_edges*max_task_length).astype(int)
            task_gt = pd.Series(3, index=list(range(max_task_length)))
            task_gt.loc[task_edges[0]:task_edges[1]-1] = 0
            task_gt.loc[task_edges[1]:task_edges[2]] = 1
            task_gt.loc[task_edges[2]+1:task_edges[3]] = 2
            tasks_gt.append(task_gt)
        return tasks_gt

#%%
fold = 10
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import time
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn

plt.figure()

class deep_MTL_UNet(nn.Module):
    def __init__(self):
        super(deep_MTL_UNet, self).__init__()
        self.down0 = nn.Sequential(nn.Conv1d(12, 64, 3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv1d(64, 64, 3, padding=1),
                                   nn.ReLU())
        output_length = kernel1d_output_length(max_task_length, 3, 1, 1)
        output_length = kernel1d_output_length(output_length, 3, 1, 1)
        
        self.down1 = nn.Sequential(nn.MaxPool1d(2, 2),
                                   nn.Conv1d(64, 128, 3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv1d(128, 128, 3, padding=1),
                                   nn.ReLU())
        output_length = kernel1d_output_length(output_length, 2, 2)
        output_length = kernel1d_output_length(output_length, 3, 1, 1)
        output_length = kernel1d_output_length(output_length, 3, 1, 1)

        self.down2 = nn.Sequential(nn.MaxPool1d(2, 2),
                                   nn.Conv1d(128, 256, 3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv1d(256, 256, 3, padding=1),
                                   nn.ReLU())
        output_length = kernel1d_output_length(output_length, 2, 2)
        output_length = kernel1d_output_length(output_length, 3, 1, 1)
        output_length = kernel1d_output_length(output_length, 3, 1, 1)

        self.down3 = nn.Sequential(nn.MaxPool1d(2, 2),
                                   nn.Conv1d(256, 512, 3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv1d(512, 512, 3, padding=1),
                                   nn.ReLU())
        output_length = kernel1d_output_length(output_length, 2, 2)
        output_length = kernel1d_output_length(output_length, 3, 1, 1)
        output_length = kernel1d_output_length(output_length, 3, 1, 1)
        
        self.down4 = nn.Sequential(nn.MaxPool1d(2, 2),
                                   nn.Conv1d(512, 1024, 3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv1d(1024, 1024, 3, padding=1),
                                   nn.ReLU())
        output_length = kernel1d_output_length(output_length, 2, 2)
        output_length = kernel1d_output_length(output_length, 3, 1, 1)
        output_length = kernel1d_output_length(output_length, 3, 1, 1)

        self.turn1 = nn.Sequential(nn.MaxPool1d(2, 2),
                                   nn.Conv1d(1024, 2048, 3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv1d(2048, 2048, 3, padding=1),
                                   nn.ReLU())
        output_length = kernel1d_output_length(output_length, 2, 2)
        output_length = kernel1d_output_length(output_length, 3, 1, 1)
        output_length = kernel1d_output_length(output_length, 3, 1, 1)
        
        self.turn2 = nn.ConvTranspose1d(2048, 1024, 2, 2)

        self.up1 = nn.Sequential(nn.Conv1d(2048, 1024, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv1d(1024, 1024, 3, padding=1),
                                 nn.ReLU(),
                                 nn.ConvTranspose1d(1024, 512, 2, 2))

        self.up2 = nn.Sequential(nn.Conv1d(1024, 512, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv1d(512, 512, 3, padding=1),
                                 nn.ReLU(),
                                 nn.ConvTranspose1d(512, 256, 2, 2))
 
        self.up3 = nn.Sequential(nn.Conv1d(512, 256, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv1d(256, 256, 3, padding=1),
                                 nn.ReLU(),
                                 nn.ConvTranspose1d(256, 128, 2, 2))

        self.up4 = nn.Sequential(nn.Conv1d(256, 128, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv1d(128, 128, 3, padding=1),
                                 nn.ReLU(),
                                 nn.ConvTranspose1d(128, 64, 2, 2))

        self.last = nn.Sequential(nn.Conv1d(128, 64, 3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv1d(64, 64, 3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv1d(64, 4, 1))

        self.MLP = nn.Sequential(nn.Linear(output_length*2048, 2**10),
                                 nn.Linear(2**10, 4))
        
    def forward(self, x):
        x_down0 = self.down0(x.float())
        x_down1 = self.down1(x_down0)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x_down4 = self.down4(x_down3)
        
        x_turn1 = self.turn1(x_down4)
        x_turn2 = self.turn2(x_turn1)

        x_up1 = self.up1(torch.cat((x_down4, padding_check(x_down4, x_turn2)), dim=1))
        x_up2 = self.up2(torch.cat((x_down3, padding_check(x_down3, x_up1)), dim=1))
        x_up3 = self.up3(torch.cat((x_down2, padding_check(x_down2, x_up2)), dim=1))
        x_up4 = self.up4(torch.cat((x_down1, padding_check(x_down1, x_up3)), dim=1))
        x_last = self.last(torch.cat((x_down0, padding_check(x_down0, x_up4)), dim=1))
        
        x_MLP = self.MLP(torch.flatten(x_turn1, start_dim=1))
        
        return x_last, x_MLP

def padding_check(tensor_down, tensor_up):
    if tensor_down.size(dim=2)%2==1:
        return torch.cat((tensor_up, torch.zeros(tensor_up.size(dim=0), tensor_up.size(dim=1), 1).to(config_sys['device'])), dim=2)
    else:
        return tensor_up

def kernel1d_output_length(L_in, kernel_size, stride, padding=0, dialation=1):
    return int((L_in + 2*padding - dialation*(kernel_size - 1) -1)/stride + 1)

def train(model: nn.Module, dataloader):
    model.train()
    outputs_gt = []
    outputs_edges = []
    train_clf_losses = []
    train_reg_losses = []
    for x, y in dataloader:
        data, targets_gt, targets_edges = x.to(config_sys['device']), y[:, 4:].to(config_sys['device']), y[:, 0:4].to(config_sys['device'])
        
        output_gt, output_edges = model(data)
        loss1 = config_sys['clf_loss_weight']*criterion1(output_gt, targets_gt.long()) 
        loss2 = config_sys['reg_loss_weight']*criterion2(output_edges, targets_edges.float())
        
        optimizer.zero_grad()
        (loss1+loss2).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        train_clf_losses.append(loss1.item())
        train_reg_losses.append(loss2.item())
        
        softmax = nn.Softmax(dim=1)
        output_gt = softmax(output_gt).cpu().detach().numpy()
        output_edges = output_edges.cpu().detach().numpy()
        for index in range(np.shape(output_gt)[0]):
            outputs_gt.append(np.argmax(output_gt[index, :, :], axis=0))
            outputs_edges.append(output_edges[index, :])
        
    train_clf_loss = np.mean(train_clf_losses)
    train_reg_loss = np.mean(train_reg_losses)
        
    return outputs_gt, outputs_edges, train_clf_loss, train_reg_loss

def test(model: nn.Module, dataloader):
    model.eval()
    outupts_gt = []
    outputs_edges = []
    test_clf_losses = []
    test_reg_losses = []
    with torch.no_grad():
        for x, y in dataloader:
            data, targets_gt, targets_edges = x.to(config_sys['device']), y[:, 4:].to(config_sys['device']), y[:, 0:4].to(config_sys['device'])
            
            output_gt, output_edges = model(data)
            loss1 = config_sys['clf_loss_weight']*criterion1(output_gt, targets_gt.long())
            loss2 = config_sys['reg_loss_weight']*criterion2(output_edges, targets_edges.float())
            
            test_clf_losses.append([loss1.item()])
            test_reg_losses.append([loss2.item()])
            
            softmax = nn.Softmax(dim=1)
            output_gt = softmax(output_gt).cpu().detach().numpy()
            output_edges = output_edges.cpu().detach().numpy()
            for index in range(np.shape(output_gt)[0]):
                outupts_gt.append(np.argmax(output_gt[index, :, :], axis=0))
                outputs_edges.append(output_edges[index, :])

    test_clf_loss = np.mean(test_clf_losses)
    test_reg_loss = np.mean(test_reg_losses)
    
    return outupts_gt, outputs_edges, test_clf_loss, test_reg_loss

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
def WarmUp(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, 
           num_cycles: float = 0.5, last_epoch: int = -1,):
  def lr_lambda(current_step):
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step-num_warmup_steps)/float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5*(1.0+math.cos(math.pi*float(num_cycles)*2.0*progress)))
  return LambdaLR(optimizer, lr_lambda, last_epoch)


def plot_clf_learning_curve(train_losses, test_losses, plt_name):
    plt.figure()
    pd.Series(train_losses).plot()
    pd.Series(test_losses).plot()
    plt.title(plt_name)
    plt.xlabel('Epoch index')
    plt.ylabel('Loss')
    plt.legend(['train loss', 'test loss'])
    plt.show()
    plt.clf()
    plt.close()

def plot_reg_learning_curve(train_losses, test_losses, plt_name):
    plt.figure()
    pd.Series(train_losses).plot(ylim=(0, 0.0025))
    pd.Series(test_losses).plot(ylim=(0, 0.0025))
    plt.title(plt_name)
    plt.xlabel('Epoch index')
    plt.ylabel('Loss')
    plt.legend(['train loss', 'test loss'])
    plt.show()
    plt.clf()
    plt.close()
    

def plot_performance(plt_name, metrics_table_fold, f1_table_fold, truth, prediction, normalize=None):
    '''normalize={‘true’, ‘pred’, ‘all’}, default=None'''
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios':[1, 1, 5]})
    fig.suptitle(plt_name)
    
    pd.plotting.table(ax=ax1, data=metrics_table_fold, loc='center')
    ax1.axis('off')
    
    pd.plotting.table(ax=ax2, data=f1_table_fold, loc='center')
    ax2.axis('off')
    
    matrix = metrics.confusion_matrix(truth, prediction, normalize=normalize)
    sn.heatmap(matrix, cmap ='coolwarm', linecolor='white', linewidths=1, annot=True, fmt='g', ax=ax3)
    ax3.set(xlabel='Prediction', ylabel='Truth')

    plt.plot()
    plt.show()
    plt.clf()
    plt.close()

tasks_label = []
for task_index in range(len(tasks_gt)):
    tasks_label.append(pd.concat([tasks_edges[task_index], tasks_gt[task_index]], ignore_index=True))

outputs_clf_gt = [[]]*len(tasks)
outputs_reg_gt = [[]]*len(tasks)
metrics_clf_gt_table = pd.DataFrame(0, columns=['Recall', 'Precision', 'F1-score'], index=list(range(1,fold+1)))
metrics_reg_gt_table = pd.DataFrame(0, columns=['Recall', 'Precision', 'F1-score'], index=list(range(1,fold+1)))
kf = KFold(fold, shuffle=True, random_state=0)
for fold_index, (train_indexes, test_indexes) in enumerate(kf.split(tasks)):
    fold_index += 1
    print('\nFold', fold_index)
    
    train_tasks = []
    train_tasks_label = []
    train_tasks_gt = []
    train_tasks_edges = []
    for train_index in train_indexes:
        train_tasks.append(tasks[train_index])
        train_tasks_label.append(tasks_label[train_index])
        train_tasks_gt.append(np.array(tasks_gt[train_index]))
        train_tasks_edges.append(np.array(tasks_edges[train_index]))
    train_tasks = np.array(train_tasks).astype(float).transpose(0, 2, 1)
    train_tasks_label = np.array(train_tasks_label).astype(float)
    train_set = TensorDataset(torch.from_numpy(train_tasks).to(config_sys['device']), torch.from_numpy(train_tasks_label).to(config_sys['device']))
    train_dataloader = DataLoader(train_set, batch_size=config_sys['batch_size'], shuffle=True)
    
    test_tasks = []
    test_tasks_label = []
    test_tasks_gt = []
    test_tasks_edges = []
    for test_index in test_indexes:
        test_tasks.append(tasks[test_index])
        test_tasks_label.append(tasks_label[test_index])
        test_tasks_gt.append(np.array(tasks_gt[test_index]))
        test_tasks_edges.append(np.array(tasks_edges[test_index]))
    test_tasks = np.array(test_tasks).astype(float).transpose(0, 2, 1)
    test_tasks_label = np.array(test_tasks_label).astype(float)
    test_set = TensorDataset(torch.from_numpy(test_tasks).to(config_sys['device']), torch.from_numpy(test_tasks_label).to(config_sys['device']))
    test_dataloader = DataLoader(test_set, batch_size=config_sys['batch_size'], shuffle=False)
    
    model = deep_MTL_UNet().to(config_sys['device'])
    criterion1 = nn.CrossEntropyLoss().to(config_sys['device'])
    criterion2 = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = WarmUp(optimizer, 8, config_sys['epochs'])
    
    train_tasks_gt_concat = list(np.hstack(train_tasks_gt))
    test_tasks_gt_concat = list(np.hstack(test_tasks_gt))
    train_tasks_edges_concat = np.hstack(train_tasks_edges)
    test_tasks_edges_concat = np.hstack(test_tasks_edges)
    
    train_clf_losses = []
    train_reg_losses = []
    test_clf_losses = []
    test_reg_losses = []
    for epoch in range(config_sys['epochs']):
        start_time = time.time()
        
        train_outputs_clf_gt, train_outputs_edges, train_clf_loss, train_reg_loss = train(model, train_dataloader)
        test_outputs_clf_gt, test_outputs_edges, test_clf_loss, test_reg_loss = test(model, test_dataloader)
        
        train_outputs_clf_gt_concat = list(np.hstack(train_outputs_clf_gt))
        test_outputs_clf_gt_concat = list(np.hstack(test_outputs_clf_gt))
        train_clf_f1 = metrics.f1_score(train_outputs_clf_gt_concat, train_tasks_gt_concat, average=None)
        test_clf_f1 = metrics.f1_score(test_outputs_clf_gt_concat, test_tasks_gt_concat, average=None)
        
        train_outputs_reg_gt = task_edges_to_task_gt(train_outputs_edges, max_task_length)
        test_outputs_reg_gt = task_edges_to_task_gt(test_outputs_edges, max_task_length)
        train_outputs_reg_gt_concat = list(np.hstack(train_outputs_reg_gt))
        test_outputs_reg_gt_concat = list(np.hstack(test_outputs_reg_gt))
        train_reg_f1 = metrics.f1_score(train_outputs_reg_gt_concat, train_tasks_gt_concat, average=None)
        test_reg_f1 = metrics.f1_score(test_outputs_reg_gt_concat, test_tasks_gt_concat, average=None)
        
        train_outputs_edges_concat = np.hstack(train_outputs_edges)
        test_outputs_edges_concat = np.hstack(test_outputs_edges)
        train_MAE = np.mean(np.reshape(abs((train_tasks_edges_concat*max_task_length).astype(int) - (train_outputs_edges_concat*max_task_length).astype(int)), [-1, 4]), axis=0)
        test_MAE = np.mean(np.reshape(abs((test_tasks_edges_concat*max_task_length).astype(int) - (test_outputs_edges_concat*max_task_length).astype(int)), [-1, 4]), axis=0)
        
        train_clf_losses.append(train_clf_loss)
        train_reg_losses.append(train_reg_loss)
        test_clf_losses.append(test_clf_loss)
        test_reg_losses.append(test_reg_loss)
        
        elapsed = time.time() - start_time
        
        print(f'|||| Epoch {epoch+1:3d} of fold {fold_index:2d} | time: {elapsed:3.2f}s ||||\n'
              f'| train clf loss {train_clf_loss:4.4f} | train reg loss {train_reg_loss:4.4f} |\n'
              f'| train clf f1 {np.round(train_clf_f1, 4)} |\n'
              f'| train reg f1 {np.round(train_reg_f1, 4)} |\n'
              f'| test clf loss {test_clf_loss:4.4f} | test reg loss {test_reg_loss:4.4f} |\n'
              f'| test clf f1 {np.round(test_clf_f1, 4)} |\n'
              f'| test reg f1 {np.round(test_reg_f1, 4)}|\n')

        scheduler.step()
        # scheduler.step(test_f1)

    metrics_clf_gt_table.at[fold_index, 'Recall'] = metrics.recall_score(test_outputs_clf_gt_concat, test_tasks_gt_concat, average='macro')
    metrics_clf_gt_table.at[fold_index, 'Precision'] = metrics.precision_score(test_outputs_clf_gt_concat, test_tasks_gt_concat, average='macro')
    metrics_clf_gt_table.at[fold_index, 'F1-score'] = metrics.f1_score(test_outputs_clf_gt_concat, test_tasks_gt_concat, average='macro')
    print(metrics_clf_gt_table)
    
    print()
    
    metrics_reg_gt_table.at[fold_index, 'Recall'] = metrics.recall_score(test_outputs_reg_gt_concat, test_tasks_gt_concat, average='macro')
    metrics_reg_gt_table.at[fold_index, 'Precision'] = metrics.precision_score(test_outputs_reg_gt_concat, test_tasks_gt_concat, average='macro')
    metrics_reg_gt_table.at[fold_index, 'F1-score'] = metrics.f1_score(test_outputs_reg_gt_concat, test_tasks_gt_concat, average='macro')
    print(metrics_reg_gt_table)
    
    plot_clf_learning_curve(train_clf_losses, test_clf_losses, 'Clf Loss of Fold '+str(fold_index))
    plot_reg_learning_curve(train_reg_losses, test_reg_losses, 'Reg Loss of Fold '+str(fold_index))

    plot_performance('Clf Test Performance of Fold '+str(fold_index),
                     pd.DataFrame(list(round(metrics_clf_gt_table.loc[fold_index, :]*100, 2)), index=['Recall (%)', 'Precision (%)', 'F1-score (%)']).T,
                     pd.DataFrame(list(np.around(metrics.f1_score(test_outputs_clf_gt_concat, test_tasks_gt_concat, average=None)*100, 2)), columns=['F1-score (%)'], index=['Sub-task 1', 'Sub-task 2', 'Sub-task 3', 'Zero Padding']).T,
                     test_outputs_clf_gt_concat, test_tasks_gt_concat)
    
    plot_performance('Reg Test Performance of Fold '+str(fold_index),
                     pd.DataFrame(list(round(metrics_reg_gt_table.loc[fold_index, :]*100, 2)), index=['Recall (%)', 'Precision (%)', 'F1-score (%)']).T,
                     pd.DataFrame(list(np.around(metrics.f1_score(test_outputs_reg_gt_concat, test_tasks_gt_concat, average=None)*100, 2)), columns=['F1-score (%)'], index=['Sub-task 1', 'Sub-task 2', 'Sub-task 3', 'Zero Padding']).T,
                     test_outputs_reg_gt_concat, test_tasks_gt_concat)
    
    for task_index, test_index in enumerate(test_indexes):
        outputs_clf_gt[test_index] = test_outputs_clf_gt[task_index]
        outputs_reg_gt[test_index] = test_outputs_reg_gt[task_index]
        
metrics_clf_gt_table.loc['mean', :] = metrics_clf_gt_table.loc[1:fold, :].mean()
metrics_clf_gt_table.loc['std', :] = metrics_clf_gt_table.loc[1:fold, :].std()
avg_clf_gt = [str(round(m*100, 2))+' ± '+str(round(s*100, 2)) for m, s in zip(metrics_clf_gt_table.loc['mean', :], metrics_clf_gt_table.loc['std', :])]

metrics_reg_gt_table.loc['mean', :] = metrics_reg_gt_table.loc[1:fold, :].mean()
metrics_reg_gt_table.loc['std', :] = metrics_reg_gt_table.loc[1:fold, :].std()
avg_reg_gt = [str(round(m*100, 2))+' ± '+str(round(s*100, 2)) for m, s in zip(metrics_reg_gt_table.loc['mean', :], metrics_reg_gt_table.loc['std', :])]

#%%
for task_index in range(len(tasks_gt)):
    tasks_gt[task_index] = np.array(tasks_gt[task_index])
    
outputs_clf_concat = list(np.hstack(outputs_clf_gt))
outputs_reg_concat = list(np.hstack(outputs_reg_gt))
tasks_gt_concat = list(np.hstack(tasks_gt))

plot_performance('Clf Test Performance of All Folds',
                 pd.DataFrame(avg_clf_gt, index=['Recall (%)', 'Precision (%)', 'F1-score (%)']).T,
                 pd.DataFrame(list(np.around(metrics.f1_score(outputs_clf_concat, tasks_gt_concat, average=None)*100, 2)), columns=['F1-score (%)'], index=['Sub-task 1', 'Sub-task 2', 'Sub-task 3', 'Zero Padding']).T,
                 outputs_clf_concat, tasks_gt_concat)

plot_performance('Reg Test Performance of All Folds',
                 pd.DataFrame(avg_reg_gt, index=['Recall (%)', 'Precision (%)', 'F1-score (%)']).T,
                 pd.DataFrame(list(np.around(metrics.f1_score(outputs_reg_concat, tasks_gt_concat, average=None)*100, 2)), columns=['F1-score (%)'], index=['Sub-task 1', 'Sub-task 2', 'Sub-task 3', 'Zero Padding']).T,
                 outputs_reg_concat, tasks_gt_concat)
