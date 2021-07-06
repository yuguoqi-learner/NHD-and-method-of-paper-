

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
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
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
from sklearn import preprocessing


def get_X_Y():
    path = '/home/thinkstation/YU/yuguoqi/train_datatas/'
    path_label = '/home/thinkstation/YU/yuguoqi/train_label/'
    files =os.listdir(path)
    files.sort()
    files_label = os.listdir(path_label)
    files_label.sort()
    list_X = []
    list_Y = []
    for file in files:
        if not  os.path.isdir(path+file):
            f_name = str(file)
            filename = path + f_name
            data = np.load(filename,allow_pickle=True)
            datas = data['arr_0']
            list_X.append(datas)
            y = [''.join(list(g)) for k, g in groupby(f_name, key=lambda x: x.isdigit())]
            Y = y[0]
            for file_label in files_label:
                labels = os.path.splitext(file_label)[0]
                if labels == Y:
                    Label = np.loadtxt(path_label+str(file_label))
                    # Labels = Label.transpose()
                    list_Y.append(Label)
                    break;
    Y = np.array(list_Y)

    X = np.array(list_X).astype(float)
    X_mean = np.mean(X, axis=3)
    X_std = np.std(X, axis=3)

    for j in range(2):
        for i in range(23):
            for k in range(300):
                X[:,j,i,k] = (X[:, j, i,k] - X_mean[:,j,i])

    for l in range(2):
        for m in range(23):
            for n in range(300):
                X[:,l,m,n] = X[:,l,m,n] / X_std[:,l,m]

    return X,Y
# a,b = get_X_Y()

def split_dataloader(X,Y,batch_size):
    # encoder = LabelEncoder()
    # encoder.fit(Y)
    # Y = encoder.transform(Y)
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    # X = torch.from_numpy(X)

    # Y = torch.from_numpy(Y)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

    # split = StratifiedShuffleSplit(n_splits=1,train_size=0.8, test_size=0.2, random_state=0)
    # for train_index, test_index in split.split(X, Y):
    #     X_train,Y_train= X[train_index,:, :],Y[train_index,:]
    #     X_test,Y_test = X[test_index,:, :],Y[test_index,:]

    train_dataset = TensorDataset(X_train,Y_train)
    test_dataset = TensorDataset(X_test,Y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader,test_loader

def encoder_trans(Y,num_class):
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    num_class = torch.max(num_class, 1)[1]
    y_pred = num_class.detach().numpy()
    label = encoder.inverse_transform(y_pred)
    return label


def random_dataloader(X,Y,batch_size):
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
    train_dataset = TensorDataset(X_train,Y_train)
    test_dataset = TensorDataset(X_test,Y_test)
    train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True,)
    test_loader = DataLoader(dataset = test_dataset,batch_size = batch_size,shuffle= True)
    return train_loader,test_loader