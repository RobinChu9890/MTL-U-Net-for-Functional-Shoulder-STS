#%% Set path for retrieving the trained model and the validation set

model_path = r''
val_set_path = r''

#%% Import the validation set and wrap the data with dataloader

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

val_set = (np.load(val_set_path, allow_pickle=True)).item()
max_task_length = len(val_set['N01_WH']['data'])
val_datas = []
val_labels = []
val_gts = []
val_edges = []
for key in list(val_set.keys()):
    val_datas.append(val_set[key]['data'])
    val_labels.append(pd.concat([val_set[key]['edge'], val_set[key]['gt']], ignore_index=True))
    val_gts.append(val_set[key]['gt'])
    val_edges.append(val_set[key]['edge'])
val_datas = np.array(val_datas).astype(float).transpose(0, 2, 1)
val_labels = np.array(val_labels).astype(float)
val_dataset = TensorDataset(torch.from_numpy(val_datas).to(device), torch.from_numpy(val_labels).to(device))
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

#%% Import the trained model and identify the validation data with the model

model = torch.jit.load(model_path)
model.eval()

output_sts_gts = []
output_tpd_edges = []
with torch.no_grad():
    for x, y in val_dataloader:
        data, target_gts, target_edges = x.to(device), y[:, 4:].to(device), y[:, 0:4].to(device)
        
        output_sts_gt, output_tpd_edge = model(data)

        softmax = nn.Softmax(dim=1)
        output_sts_gt = softmax(output_sts_gt).cpu().detach().numpy()
        output_tpd_edge = output_tpd_edge.cpu().detach().numpy()
        for index in range(np.shape(output_sts_gt)[0]):
            output_sts_gts.append(np.argmax(output_sts_gt[index, :, :], axis=0))
            output_tpd_edges.append(output_tpd_edge[index, :])

#%% Evaluate the perfomance of the outputted ground truth and timestamps of edges

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn

def edges_to_gt(tasks_edges, max_task_length):
    tasks_gt = []
    for task_edges in tasks_edges:
        task_edges = (task_edges*max_task_length).astype(int)
        task_gt = pd.Series(0, index=list(range(max_task_length)))
        task_gt.loc[task_edges[0]:task_edges[1]-1] = 1
        task_gt.loc[task_edges[1]:task_edges[2]] = 2
        task_gt.loc[task_edges[2]+1:task_edges[3]] = 3
        tasks_gt.append(task_gt)
    return tasks_gt

def plot_performance(plt_name, metrics_table_fold, f1_table_fold, truth, prediction, normalize=None):
    '''normalize={‘true’, ‘pred’, ‘all’}, default=None'''
    plt.figure()  
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios':[1, 1, 8]})
    
    pd.plotting.table(ax=ax1, data=metrics_table_fold, loc='center')
    ax1.set_title(plt_name+' (with average method as Macro)')
    ax1.axis('off')
    
    pd.plotting.table(ax=ax2, data=f1_table_fold, loc='center')
    ax2.set_title('F1-score of each class')
    ax2.axis('off')
    
    matrix = metrics.confusion_matrix(truth, prediction, normalize=normalize)
    sn.heatmap(matrix, cmap ='coolwarm', linecolor='white', linewidths=1, annot=True, fmt='g', ax=ax3)
    ax3.set_title('Confusion matrix')
    ax3.set_xticklabels(['Zero Padding', 'Sub-task 1', 'Sub-task 2', 'Sub-task 3'], fontsize=8)
    ax3.set_yticklabels(['Zero Padding', 'Sub-task 1', 'Sub-task 2', 'Sub-task 3'], fontsize=8)
    ax3.set(xlabel='Prediction', ylabel='Truth')

    plt.plot()
    plt.show()
    plt.clf()
    plt.close()

output_tpd_gts = edges_to_gt(output_tpd_edges, max_task_length)

output_sts_gts_concat = list(np.hstack(output_sts_gts))
output_tpd_gts_concat = list(np.hstack(output_tpd_gts))
val_gts_concat = list(np.hstack(val_gts))

performance_table_sts = pd.DataFrame(0, columns=['Recall', 'Precision', 'F1-score'], index=[0])
performance_table_tpd = pd.DataFrame(0, columns=['Recall', 'Precision', 'F1-score'], index=[0])

performance_table_sts.at[0, 'Recall'] = metrics.recall_score(val_gts_concat, output_sts_gts_concat, average='macro')
performance_table_sts.at[0, 'Precision'] = metrics.precision_score(val_gts_concat, output_sts_gts_concat, average='macro')
performance_table_sts.at[0, 'F1-score'] = metrics.f1_score(val_gts_concat, output_sts_gts_concat, average='macro')

performance_table_tpd.at[0, 'Recall'] = metrics.recall_score(val_gts_concat, output_tpd_gts_concat, average='macro')
performance_table_tpd.at[0, 'Precision'] = metrics.precision_score(val_gts_concat, output_tpd_gts_concat, average='macro')
performance_table_tpd.at[0, 'F1-score'] = metrics.f1_score(val_gts_concat, output_tpd_gts_concat, average='macro')

plot_performance('Performance of STS',
                 pd.DataFrame(list(round(performance_table_sts.loc[0, :]*100, 2)), index=['Recall (%)', 'Precision (%)', 'F1-score (%)']).T,
                 pd.DataFrame(list(np.around(metrics.f1_score(val_gts_concat, output_sts_gts_concat, average=None)*100, 2)), columns=['F1-score (%)'], index=['Zero Padding', 'Sub-task 1', 'Sub-task 2', 'Sub-task 3']).T,
                 val_gts_concat, output_sts_gts_concat)

plot_performance('Performance of TPD',
                 pd.DataFrame(list(round(performance_table_tpd.loc[0, :]*100, 2)), index=['Recall (%)', 'Precision (%)', 'F1-score (%)']).T,
                 pd.DataFrame(list(np.around(metrics.f1_score(val_gts_concat, output_tpd_gts_concat, average=None)*100, 2)), columns=['F1-score (%)'], index=['Zero Padding', 'Sub-task 1', 'Sub-task 2', 'Sub-task 3']).T,
                 val_gts_concat, output_tpd_gts_concat)
