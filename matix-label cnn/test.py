
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
# from functions import *
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
from torch.utils.data import DataLoader, TensorDataset, Dataset
from itertools import groupby
import heapq
import matplotlib.pyplot as plt
import random
from torch.backends import cudnn
import seaborn as sn
import pandas as pd
from torch.optim import lr_scheduler
from random_seed import seed
from dataset import get_X_Y,random_dataloader,split_dataloader,encoder_trans
from model import LeNet5,Classifier,Resnet
from device import  device

def max_index_pred(data):
    list_index = []  # 创建列表,存放最大值的索引
    # data = data.detach().cpu().numpy()

    for i in range(len(data)):
        datas = data[i]
        dim = datas.ravel()
        nums_max = heapq.nlargest(1, range(len(dim)), dim.take)
        num_max_index = []
        for j in nums_max:
            rowindex = j // 10  # 0 start
            colindex = j % 10
        #     max_index = [rowindex,colindex]
        #     num_max_index.append(max_index)
        # ave_row_index = (num_max_index[0][0]+num_max_index[1][0]+num_max_index[2][0])/len(num_max_index)
        # ave_col_index = (num_max_index[0][1]+num_max_index[1][1]+num_max_index[2][1])/len(num_max_index)
        index = tuple((i,rowindex,colindex))
        list_index.append(index)
    return list_index


def acc(y,pred):
    num = 0
    y,pred = np.array(y),np.array(pred)
    if np.sqrt(np.sum(np.power((y - pred), 2))) == np.sqrt(0):
        num = 1
    return num

batchs_acc = []
list_y = []
list_pred = []
batchs_acc_e = []
batchs_acc_h = []
list_pred_e = []
list_pred_h = []
def test(data,y):
    lenet5, classifier =LeNet5(),Classifier()
    # lstm.eval()
    lenet5.eval()
    classifier.eval()

    #Load param
    # lstm.load_state_dict(torch.load("/home/thinkstation/YU/yuguoqi/model_save/lstm_epoch130.pth"))
    lenet5.load_state_dict(torch.load("/home/thinkstation/YU/yuguoqi/model_save/cnn72.pth"))
    classifier.load_state_dict(torch.load("/home/thinkstation/YU/yuguoqi/model_save/classifier_epoch72.pth"))
    # optimizer.load_state_dict(torch.load("model_save/optimizer_epoch200.pth"))

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")

    lenet5.to(device)
    # lstm.to(device)
    classifier.to(device)

    data = data.float().to(device)
    # data= data.transpose(2,3)

    output_cnn = lenet5(data)
    output = classifier(output_cnn)
    output=output.cpu().detach().numpy()

    y = y[np.newaxis,:,:]
    y_pred = max_index_pred(output)
    y = max_index_pred(y)

    y_e = y[0][1]
    y_h = y[0][2]
    y_pred_e = y_pred[0][1]
    y_pred_h = y_pred[0][2]

    # batch_acc = acc(y_pred, y)
    # batchs_acc.append(batch_acc)

    batch_acc_e = acc(y_e, y_pred_e)
    batch_acc_h = acc(y_h, y_pred_h)
    batchs_acc_e.append(batch_acc_e)
    batchs_acc_h.append(batch_acc_h)
    #
    # list_y.append(np.array(y))
    list_pred.append(np.array(y_pred))
    list_pred_e.append(y_pred_e)
    list_pred_h.append(y_pred_h)

    return list_pred,list_pred_e,list_pred_h,batchs_acc_e,batchs_acc_h

def get_data_label():
    path = '/home/thinkstation/YU/yuguoqi/test_datatas/'
    path_label = '/home/thinkstation/YU/yuguoqi/test_label/'
    files =os.listdir(path)
    files.sort()
    files_label = os.listdir(path_label)
    files_label.sort()
    list_X = []
    list_Y = []
    name_list = []
    for file in files:
        if not os.path.isdir(path + file):
            f_name = str(file)
            filename = path + f_name
            data = np.load(filename, allow_pickle=True)
            datas = data['arr_0']
            list_X.append(datas)
            y = [''.join(list(g)) for k, g in groupby(f_name, key=lambda x: x.isdigit())]
            Y = y[0]
            for file_label in files_label:
                labels = os.path.splitext(file_label)[0]
                if labels == Y:
                    Label = np.loadtxt(path_label + str(file_label))
                    name = file_label
                    name_list.append(name)
                    # Labels = Label.transpose()
                    list_Y.append(Label)
                    break;
    X = np.array(list_X)
    X_mean = np.mean(X, axis=3)
    X_std = np.std(X, axis=3)
    for j in range(2):
        for i in range(23):
            for k in range(300):
                X[:, j, i, k] = (X[:, j, i, k] - X_mean[:, j, i])

    for l in range(2):
        for m in range(23):
            for n in range(300):
                X[:, l, m, n] = X[:, l, m, n] / X_std[:, l, m]
    Y = np.array(list_Y)

    # X = list_X
    # Y = list_Y
    return X, Y,name_list

datas,label,name_list = get_data_label()

for i in range(len(datas)):
    datass = datas[i]
    labels = label[i]
    datass = torch.from_numpy(datass)
    datasss = datass.unsqueeze(0)
    list_pred,list_pred_e,list_pred_h,batchs_acc_e,batchs_acc_h = test(datasss,labels)
# class_acc = sum(class_num)/len(class_num)
a = list_pred_e[0:50]
b = list_pred_h[50:100]
c = list_pred_h[100:150]
d = list_pred_h[150:200]
e = list_pred_h[200:250]
f = list_pred_h[250:300]
a.sort()
b.sort()
c.sort()
d.sort()
e.sort()
f.sort()

# np.savetxt('/home/thinkstation/YU/Badminton_h',a)
# np.savetxt('/home/thinkstation/YU/BigDrawPaper_h.txt',b)
# np.savetxt('/home/thinkstation/YU/Earmuffs_h.txt',c)
# np.savetxt('/home/thinkstation/YU/HDD_h.txt',d)
# np.savetxt('/home/thinkstation/YU/InkpadBox_h.txt',e)
# np.savetxt('/home/thinkstation/YU/NailBox_h.txt',f)

Badminton_e = sum(list_pred_e[0:50])/50
BigDrawPaper_e = sum(list_pred_e[50:100])/50
Earmuffs_e = sum(list_pred_e[100:150])/50
HDD_e =sum(list_pred_e[150:200])/50
InkpadBox_e = sum(list_pred_e[200:250])/50
NailBox_e = sum(list_pred_e[250:300])/50

Badminton_h = sum(list_pred_h[0:50])/50
BigDrawPaper_h = sum(list_pred_h[50:100])/50
Earmuffs_h = sum(list_pred_h[100:150])/50
HDD_h =sum(list_pred_h[150:200])/50
InkpadBox_h = sum(list_pred_h[200:250])/50
NailBox_h = sum(list_pred_h[250:300])/50

Badminton_acc_e = sum(batchs_acc_e[0:50])/50
BigDrawPaper_acc_e = sum(batchs_acc_e[50:100])/50
DoublelayerFoamBoard_acc_e = sum(batchs_acc_e[100:150])/50
Earmuffs_acc_e =sum(batchs_acc_e[150:200])/50
InkpadBox_acc_e = sum(batchs_acc_e[200:250])/50
NailBox_acc_e = sum(batchs_acc_e[250:300])/50

Badminton_acc_h = sum(batchs_acc_h[0:50])/50
BigDrawPaper_acc_h = sum(batchs_acc_h[50:100])/50
DoublelayerFoamBoard_acc_h = sum(batchs_acc_h[100:150])/50
Earmuffs_acc_h =sum(batchs_acc_h[150:200])/50
InkpadBox_acc_h = sum(batchs_acc_h[200:250])/50
NailBox_acc_h = sum(batchs_acc_h[250:300])/50

e_acc = sum(list_pred_e)/300
h_acc = sum(list_pred_h)/300
print(h_acc)
# print('len',len(class_acc))
# print('clas',class_acc)

# BlackBandage_listpred = sum(listpred[0:50])/50
# InkpadBox_listpred = sum(listpred[0:50])/50  #[7,6]
# RoundSponge_listpred = sum(listpred[50:100])/50 #[6,0]
# SoapBox_listpred = sum(listpred[100:150])/50#[5,5]
# # Tissue_listpred = sum(listpred[200:250])/50#
# WhiteThread_listpred = sum(listpred[150:-1])/50
#
# test_accu = (clas[-1]+1)/len(clas)

# print('pre_out acc rate',test_accu)




# x1, y1 = zip(*list_clas[0:50])
# x2,y2 = zip(*list_clas[50:100])
# x3,y3 = zip(*list_clas[100:150])
# x4,y4 = zip(*list_clas[150:-1])
# plt.figure()
# # plt.scatter(x1,y1,color = "#808080")
# # plt.scatter(x2,y2,color = "#666666")
# plt.scatter(x3,y3,color = "#CCCCCC")
# # plt.scatter(x4,y4,color = "#000000")
#
# plt.show()


