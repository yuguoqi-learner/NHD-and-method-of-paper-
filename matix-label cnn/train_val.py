
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
import matplotlib.pyplot as plt
import random
from torch.backends import cudnn
import seaborn as sn
import pandas as pd
from torch.optim import lr_scheduler
import heapq

def max_index_pred(data):
    list_index = []  # 创建列表,存放最大值的索引
    data = data.detach().cpu().numpy()

    # list_max =[]
    # index_maxs = [maxs.argmax() for maxs in data]
    # for index,index_max in enumerate(index_maxs):
    #     rowindex = index_max//10
    #     colindex = index_max%10
    #     list_max.append([index,rowindex,colindex])

    for i in range(len(data)):
        datas = data[i]
        dim = datas.ravel()
        nums_max = heapq.nlargest(1, range(len(dim)), dim.take)
        num_max_index = []
        for j in nums_max:
            rowindex = j // 10
            colindex = j % 10
            # max_index = [rowindex,colindex]
            # num_max_index.append(max_index)
        # ave_row_index = (num_max_index[0][0]+num_max_index[1][0]+num_max_index[2][0])/len(num_max_index)
        # ave_col_index = (num_max_index[0][1]+num_max_index[1][1]+num_max_index[2][1])/len(num_max_index)
        index = tuple((i,rowindex,colindex))
        list_index.append(index)

    # index = []
    # for i in range(len(data)):
    #     datas = data[i]
    #     dim = datas.ravel()
    #     maxs = max(dim)
    #     for j in range(len(dim)):
    #         if dim[j] == maxs:
    #             position = np.unravel_index(j,datas.shape,order='C')
    #             # position = np.array(position)
    #             tub = (i,)
    #             positions = tub+position
    #             index.append(positions)
    return list_index


def max_index_y(data):
    list_index = []
    data = data.detach().cpu().numpy()
    index_maxs = [maxs.argmax() for maxs in data]
    for index,index_max in enumerate(index_maxs):
        rowindex = index_max//10
        colindex = index_max%10
        list_index.append((index,rowindex,colindex))

    # for i in range(len(data)):
    #     datas = data[i]
    #     dim = datas.ravel()
    #     nums_max = heapq.nlargest(3, range(len(dim)), dim.take)
    #     num_max_index = []
    #     for i in nums_max:
    #         rowindex = i // 10
    #         colindex = i%10
    #         max_index = [rowindex,colindex]
    #         num_max_index.append(max_index)
    #     ave_row_index = (num_max_index[0][0]+num_max_index[1][0]+num_max_index[2][0])/len(num_max_index)
    #     ave_col_index = (num_max_index[0][1]+num_max_index[1][1]+num_max_index[2][1])/len(num_max_index)
    #     index = tuple((i,int(ave_row_index),int(ave_col_index)))

        # maxs = max(dim)
        # for j in range(len(dim)):
        #     if dim[j] == maxs:
        #         position = np.unravel_index(j,datas.shape,order='C')
        #         # position = np.array(position)
        #         tub = (i,)
        #         positions = tub+position
        #         index.append(positions)
    return  list_index


def acc(y,pred):
    num = 0
    # print('y',y)
    for i in range(len(y)):
        if y[i] == pred[i]:
            num = num+1
    acc = num/len(y)
    return acc


def train(model, device, train_loader, optimizer, loss_func,epoch):
    lenet5,classifier= model

    lenet5.train()
    classifier.train()
    losses =[]
    train_loss= []
    batchs_acc = []
    batchs_acc_e = []
    batchs_acc_h = []
    train_acc = []
    train_acc_e = []
    train_acc_h = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        X = X.to(device)
        # y = y.type(torch.long).to(device)
        y = y.to(device)

        output_cnn = lenet5(X)
        output = classifier(output_cnn)
        # output= classifier(output_lstm[0])

        # loss = loss_func(output, y.view(y.size()[0]))
        # losses.append(loss.item())
        # y_pred = torch.max(output, 1)[1]
        # y_pred = y_pred.data.cpu()
        # y = y.data.cpu()

        loss = loss_func(output, y)
        losses.append(loss.item())

        y = max_index_pred(y)
        y_pred = max_index_pred(output)
        batch_acc = acc(y,y_pred)
        batchs_acc.append(batch_acc)

        y_e = [y[i][1] for i in range(len(y))]
        y_h = [y[i][2] for i in range(len(y))]
        y_pred_e = [y_pred[i][1] for i in range(len(y_pred))]
        y_pred_h = [y_pred[i][2] for i in range(len(y_pred))]
        batch_acc_e = acc(y_e,y_pred_e)
        batch_acc_h = acc(y_h,y_pred_h)
        batchs_acc_e.append(batch_acc_e)
        batchs_acc_h.append(batch_acc_h)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss.append(np.mean(losses))
    train_acc.append(np.mean(batchs_acc))
    train_acc_e.append(np.mean(batchs_acc_e))
    train_acc_h.append(np.mean(batchs_acc_h))
    print('\ntrain set : loss: {:.4f}, Accuracy: {:.2f}%, train_acc_e:{:.2f}%, train_acc_h:{:.2f}%\n'.format(train_loss[-1],100 * train_acc[-1],100*train_acc_e[-1],100*train_acc_h[-1]))

    return train_loss,train_acc,train_acc_e,train_acc_h


def validation(model, device, test_loader,optimizer, loss_func,epoch):
    lenet5,classifier= model

    lenet5.eval()
    # resnet.eval()
    classifier.eval()
    batchs_acc = []
    test_acc = []
    test_loss = []
    losses = []
    batchs_acc_e = []
    batchs_acc_h = []
    test_acc_e = []
    test_acc_h = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            output_cnn = lenet5(X)
            output= classifier(output_cnn)
            # output = classifier(output_lstm[0])

            # output_cat = torch.cat((output_lstm,output_len),dim=1)
            # output = classifier(output_cat)
            # loss = loss_func(output, y.view(y.size()[0]))
            # losses.append(loss.item())
            # y_pred = torch.max(output, 1)[1]

            loss = loss_func(output, y)
            losses.append(loss.item())
            y_pred = max_index_pred(output)
            y = max_index_pred(y)

            batch_acc = acc(y, y_pred)
            batchs_acc.append(batch_acc)

            y_e = [y[i][1] for i in range(len(y))]
            y_h = [y[i][2] for i in range(len(y))]
            y_pred_e = [y_pred[i][1] for i in range(len(y_pred))]
            y_pred_h = [y_pred[i][2] for i in range(len(y_pred))]
            batch_acc_e = acc(y_e, y_pred_e)
            batch_acc_h = acc(y_h, y_pred_h)
            batchs_acc_e.append(batch_acc_e)
            batchs_acc_h.append(batch_acc_h)
    test_loss.append(np.mean(losses))
    test_acc.append(np.mean(batchs_acc))
    test_acc_e.append(np.mean(batchs_acc_e))
    test_acc_h.append(np.mean(batchs_acc_h))
    print('validation loss: {:.4f}, Accuracy: {:.2f}%, train_acc_e:{:.2f}%, train_acc_h:{:.2f}%\n'.format(test_loss[-1],100*test_acc[-1],100*test_acc_e[-1],100*test_acc_h[-1]))
    print("Epoch {}!".format(epoch + 1))


    # torch.save(lenet5.state_dict(),
    #            os.path.join('/home/thinkstation/YU/yuguoqi/model_save','cnn{}.pth'.format(epoch + 1)))

    # torch.save(classifier.state_dict(),
    #            os.path.join('/home/thinkstation/YU/yuguoqi/model_save','classifier_epoch{}.pth'.format(epoch + 1)))
    # torch.save(optimizer.state_dict(),
    #            os.path.join('/home/thinkstation/YU/yuguoqi/model_save','optimizer_epoch{}.pth'.format(epoch + 1)))
    # print("Epoch {} model saved!".format(epoch + 1))

    return test_loss,test_acc,test_acc_e,test_acc_h