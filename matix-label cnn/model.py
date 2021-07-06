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
from sklearn.model_selection import train_test_split
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



class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.input_dim = 79
        self.num_layers = 2
        self.hidden_dim = 512
        # self.FC_hidden_dim = 1024
        # self.FC_dim = 256
        # self.fc_dim = 10
        self.drop_p = 0.2
        # self.embed_dim = 64


        self.LSTM = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.drop_p,
            bias=True,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_`step, input_size)
            bidirectional=True  #,if true,next layer hidden_size*2
        )

        # self.fc1 = nn.Linear(self.FC_hidden_dim, self.FC_dim)
        # self.bn1 = nn.BatchNorm1d(self.FC_dim)
        # self.fc2 = nn.Linear(self.FC_dim, self.embed_dim)
        # self.bn2 = nn.BatchNorm1d(self.fc_dim, momentum=0.01)
        # self.fc3 = nn.Linear(self.fc_dim, self.embed_dim)
    def forward(self, x):
        x = self.LSTM(x)
        return x



class LeNet5(torch.nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        # self.conv11 = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1 = torch.nn.Conv2d(in_channels=2, out_channels=6, kernel_size=2, stride=1, padding=0, bias=True)
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=2, stride=1, padding=0, bias=True)
        # # Max-pooling
        # self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0,bias=True)

    def forward(self, x):
        # x = torch.nn.functional.leaky_relu(self.conv11(x))
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        # x = self.max_pool_2(x)
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = torch.squeeze(x,dim =1)
        return x

    def initialize_weights(net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 1)
                # m.weight.data.uniform_(0, 0.02)
                m.bias.data.zero_()


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=300,out_channels=40, kernel_size=1,stride=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=40, out_channels=20, kernel_size=1, stride=1),
            nn.LeakyReLU(),
        )

        self.FC1 = nn.Linear(20, 10)
        self.BN1 = nn.BatchNorm1d(10)
        # self.FC2 = nn.Linear(32, 10)
        self.drop_p = 0.5
    def forward(self, x):
        x = x.transpose(2,1)
        x = self.features(x)
        x = x.transpose(2,1)
        x = self.FC1(x)
        x = self.BN1(x)
        x = torch.sigmoid(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        # x = self.FC2(x)
        return x


    def initialize_weights(net):
        for m in net.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.FC1 = nn.Linear(148,100)
        self.BN1 = nn.BatchNorm1d(10)
        self.FC2 = nn.Linear(100, 40)
        self.BN2 = nn.BatchNorm1d(10)
        self.FC3 = nn.Linear(40, 10)
        self.drop_p = 0.2

    def forward(self, x):
        x = self.FC1(x)
        x = self.BN1(x)
        x = torch.sigmoid(x)
        x = self.FC2(x)
        x = self.BN2(x)
        x = torch.sigmoid(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.FC3(x)
        return x
    def initialize_weights(net):
        for m in net.modules():
            if isinstance(m, nn.Linear):
                # m.weight.data.uniform_(0, 0.02)
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()




class Mnasnet(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=256, drop_p=0.5, CNN_embed_dim=64):
        super(Mnasnet,self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        mnasnet = models.mnasnet1_0(pretrained=False)
        modules = list(mnasnet.children())[:-1]  # delete the last fc layer.
        self.mnasnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(12800, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x):
        # x = torch.unsqueeze(x, dim=1)
        x = self.mnasnet(x)
        x = x.view(x.size(0), -1)
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)
        return x

class Resnet(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.5, CNN_embed_dim=50):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Resnet, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.bn1(self.fc1(x))
        x = F.relu(x)
        x = self.bn2(self.fc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)
        return x




