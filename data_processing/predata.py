import numpy as np
import pandas as pd
import itertools
import random
import time

import torch 
from torch import nn
from torch.nn import functional as F
from torch import optim

# start_time = time.time()

#load data from file
forwardarray = np.fromfile("chem_forward_list.dat", dtype=int, count=-1, sep='').reshape(-1,2048)
backwardarray = np.fromfile("chem_backward_list.dat", dtype=int, count=-1, sep='').reshape(-1,2048)

# load_file = time.time()
# load_file_time = load_file - start_time

#shuffle data and change to tensor
random.seed(42)
forwardlist = forwardarray.tolist()
forwardlist_shuffle = random.sample(forwardlist,len(forwardlist))
forwardlist_shuffle.sort()
forward_list = list(forwardlist_shuffle for forwardlist_shuffle,_ in itertools.groupby(forwardlist_shuffle))
backwardlist = backwardarray.tolist()
backwardlist_shuffle = random.sample(backwardlist,len(backwardlist))
backwardlist_shuffle.sort()
backward_list = list(backwardlist_shuffle for backwardlist_shuffle,_ in itertools.groupby(backwardlist_shuffle))
forwardlist = torch.tensor(forward_list, dtype = torch.float)
backwardlist = torch.tensor(backward_list, dtype = torch.float)
#print(len(forwardlist),len(backwardlist))
#print("data prepared!")

# separate to train/val/test tensor
train_forwardlist = forwardlist[:int(0.8*len(forwardlist))].view(-1,1,2048)
train_backwardlist = backwardlist[:int(0.8*len(backwardlist))].view(-1,1,2048)
val_forwardlist = forwardlist[int(0.8*len(forwardlist)):int(0.9*len(forwardlist))].view(-1,1,2048)
val_backwardlist = backwardlist[int(0.8*len(backwardlist)):int(0.9*len(backwardlist))].view(-1,1,2048)
test_forwardlist = forwardlist[int(0.9*len(forwardlist)):].view(-1,1,2048)
test_backwardlist = backwardlist[int(0.9*len(backwardlist)):].view(-1,1,2048)

# generate_tensor = time.time()
# generate_tensor_time = generate_tensor - load_file

#add train/val/test data to dict with their labels
dict_train_x = {}
dict_train_y = {}
for i in range(len(train_forwardlist)):
    dict_train_x[i] = train_forwardlist[i]
    dict_train_y[i] = torch.ones(1)
for i in range(len(train_forwardlist),len(train_forwardlist)+len(train_backwardlist)):
    dict_train_x[i] = train_backwardlist[i-len(train_forwardlist)]
    dict_train_y[i] = torch.zeros(1)

dict_val_x = {}
dict_val_y = {}
for i in range(len(val_forwardlist)):
    dict_val_x[i] = val_forwardlist[i]
    dict_val_y[i] = torch.ones(1)
for i in range(len(val_forwardlist),len(val_forwardlist)+len(val_backwardlist)):
    dict_val_x[i] = val_backwardlist[i-len(val_forwardlist)]
    dict_val_y[i] = torch.zeros(1)

dict_test_x = {}
dict_test_y = {}
for i in range(len(test_forwardlist)):
    dict_test_x[i] = test_forwardlist[i]
    dict_test_y[i] = torch.ones(1)
for i in range(len(test_forwardlist),len(test_forwardlist)+len(test_backwardlist)):
    dict_test_x[i] = test_backwardlist[i-len(test_forwardlist)]
    dict_test_y[i] = torch.zeros(1)

# generate_dict = time.time()
# generate_dict_time = generate_dict - generate_tensor 

# build dataloader class
class Customtrainset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return dict_train_x[index],dict_train_y[index]

    def __len__(self):
        return len(dict_train_x) 

class Customvalset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return dict_val_x[index],dict_val_y[index]

    def __len__(self):
        return len(dict_val_x) 

class Customtestset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return dict_test_x[index],dict_test_y[index]

    def __len__(self):
        return len(dict_test_x) 

#print("data prepare all finished")

# generate_class = time.time()
# generate_class_time = generate_class - generate_dict 

# print("load_file_time",load_file_time)
# print("generate_tensor_time",generate_tensor_time)
# print("generate_dict_time",generate_dict_time)
# print("generate_class_time",generate_class_time)

# below are early codes or example codes, you do not need to look at them.
