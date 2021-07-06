

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
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score,confusion_matrix
import pickle
from torch.utils.data import DataLoader,TensorDataset, Dataset
from itertools import groupby
import matplotlib.pyplot as plt
import random
from torch.backends import cudnn
import seaborn as sn
import pandas as pd
from torch.optim import lr_scheduler
from random_seed import seed
from dataset import get_X_Y,random_dataloader,split_dataloader
from model import LSTM,LeNet5,Classifier,Resnet,Mnasnet,CNN1D
from train_val import train,validation
from device import device

#混淆矩阵的坐标轴
def cut_repet(Y):
    title = []
    for i in Y:
        if i not in title:
            title.append(i)
    return title


epochs = 80
batch_size = 64
learning_rate = 0.05

seed(0)

X,Y = get_X_Y()
# X = X.transpose(0,1,3,2)
train_loader,test_loader = split_dataloader(X,Y,batch_size)

# title = cut_repet(Y)
# title.sort()


use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
# device = device()


# Create model
# lstm = LSTM().to(device)

lenet5 = LeNet5().to(device)
lenet5.initialize_weights()
# cnn1d = CNN1D().to(device)
# cnn1d.initialize_weights()
classifier = Classifier().to(device)
classifier.initialize_weights()
model = [lenet5,classifier]

params =list(lenet5.parameters())+list(classifier.parameters())
optimizer = torch.optim.Adam(params, lr=learning_rate)
# torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)
# torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)

# loss_func =nn.MSELoss()
loss_func =nn.BCEWithLogitsLoss()

epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []
y = []
y_pred = []

# start training
train_loss = []
train_acc = []
train_acc_e = []
train_acc_h = []

test_loss = []
test_acc = []
test_acc_e = []
test_acc_h = []

all_y_pred = []
all_y = []
for epoch in range(epochs):
    # train, test model
    # scheduler.step()
    # lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])


    epoch_train_losses,epoch_train_scores,epoch_train_scores_e,epoch_train_scores_h = train(model, device, train_loader, optimizer,loss_func,epoch)
    epoch_test_losses,epoch_test_scores,epoch_test_scores_e,epoch_test_scores_h = validation(model, device,test_loader,optimizer,loss_func,epoch)

    train_loss.append(epoch_train_losses)
    train_acc.append(epoch_train_scores)
    train_acc_e.append(epoch_train_scores_e)
    train_acc_h.append(epoch_train_scores_h)

    test_loss.append(epoch_test_losses)
    test_acc.append(epoch_test_scores)
    test_acc_e.append(epoch_test_scores_e)
    test_acc_h.append(epoch_test_scores_h)

train_mean_10 =np.mean(train_acc[60:80])
train_mean_e_10 =np.mean(train_acc_e[60:80])
train_mean_h_10 =np.mean(train_acc_h[60:80])

test_mean_10 =np.mean(test_acc[60:80])
test_mean_e_10 =np.mean(test_acc_e[60:80])
test_mean_h_10 =np.mean(test_acc_h[60:80])
    # all_y_pred.append(y_pred)
    # all_y.append(y)
# np.savetxt('/home/thinkstation/YU/shuchushuxing_train_acc.txt',train_acc)
# np.savetxt('/home/thinkstation/YU/shuchushuxing_train_acc_e.txt',train_acc_e)
# np.savetxt('/home/thinkstation/YU/shuchushuxing_train_acc_h.txt',train_acc_h)
# np.savetxt('/home/thinkstation/YU/shuchushuxing_test_acc.txt',test_acc)
# np.savetxt('/home/thinkstation/YU/shuchushuxing_test_acc_e.txt',test_acc_e)
# np.savetxt('/home/thinkstation/YU/shuchushuxing_test_acc_h.txt',test_acc_h)

## plt.plot(lr_list)
# plt.show()

# conf_matrix = torch.zeros(num_classes,num_classes)
# output = all_y_pred[-1].tolist()
# target = all_y[-1].tolist()
# conf_matrix = confusion_matrix(output,target,conf_matrix)
#
# df_cm = pd.DataFrame(conf_matrix.numpy(),index=[i for i in list(title)],columns=[i for i in list(title)])
# plt.figure(figsize=(num_classes,num_classes))
# sn.heatmap(df_cm,annot=True,cmap="Greys")
# plt.xlabel("Labels")
# plt.ylabel("Labels-predication")
# plt.title("Confusion-matrix")
# plt.show()


plt.subplot(2,2,1)
plt.plot(train_loss[-1],color = "#000000")
plt.xlabel("num-epochs")
plt.ylabel("loss")
plt.title('Train_loss')

plt.subplot(2,2,2)
plt.plot(train_acc,color = "000000")
plt.xlabel("num-epochs")
plt.ylabel("accuracy")
plt.title('Train_acc')

plt.subplot(2,2,3)
plt.plot(test_loss,color = "000000")
plt.xlabel("num-epochs")
plt.ylabel("loss")
plt.title('Test_loss')

plt.subplot(2,2,4)
plt.plot(test_acc,color = "000000")
plt.xlabel("num-epochs")
plt.ylabel("accuracy")
plt.title('Test_acc')

plt.show()

# plt.plot(train_loss,color = "#808080",lw = 1,label = "train_loss")
# plt.plot(train_acc,color = "#00BFFF",lw = 1,label = "train_acc")
# plt.plot(train_acc_e,color = "#000000",lw = 1,label = "train_acc_e")
# plt.plot(train_acc_h,color = "#006400",lw = 1,label = "train_acc_h")

# plt.plot(test_loss,color = "#CCCCCC",lw = 1,label = "test_loss")
plt.plot(test_acc,color = "#FF69B4",lw = 1,label = "test_acc")
plt.plot(test_acc_e,color = "#6495ED",lw = 1,label = "test_acc_e")
plt.plot(test_acc_h,color = "#00008B",lw = 1,label = "test_acc_h")

plt.title("result")
plt.xlabel("num-epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()


















