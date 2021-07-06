
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
from model import LSTM,LeNet5,Classifier,Resnet,CNN1D
from train_val import train,validation



def device():
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
    return device
