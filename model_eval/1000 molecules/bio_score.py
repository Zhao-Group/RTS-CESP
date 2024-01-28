import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch 
from torch import nn
from torch.nn import functional as F
import random
from torch import optim
# import torchvision
# import torchvision.transforms as transforms
# from predata import *
# from torchmetrics import ROC
# from torchmetrics import AUROC
# from torchmetrics import PrecisionRecallCurve as PRC
# from torchmetrics import AveragePrecision as AUPRC


# train_set = Customtrainset()
# val_set = Customvalset()
# test_set = Customtestset()


# Hyper-parameters 
kernel1 = 3
kernel2 = 3
input_size = 1536
hidden_size = 512
num_classes = 1
num_epochs = 1
batch_size = 512
#val_batch_size = len(val_set)
learning_rate = 0.0001



# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_set, 
#                                            batch_size=batch_size, 
#                                            shuffle=True)

# val_loader = torch.utils.data.DataLoader(dataset=val_set, 
#                                           batch_size=val_batch_size, 
#                                           shuffle=False)

# test_loader = torch.utils.data.DataLoader(dataset=test_set, 
#                                           batch_size=batch_size, 
#                                           shuffle=False)

# values, labels = next(iter(train_loader))
# #print(values.shape, labels.shape)
# print("data loaded!")

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.convunit = nn.Sequential(
            nn.Conv1d(1, kernel1, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(kernel1),
            nn.Conv1d(kernel1, kernel2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(True),
            nn.BatchNorm1d(kernel2),
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid() 
        )
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes) 
        self.sigmoid = nn.Sigmoid() 
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.convunit(x)
        return x

bio_model = NeuralNet(input_size, hidden_size, num_classes)

# # Loss and optimizer
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  

'''
train_loss = []
val_loss = []
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (values,labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        values = values
        labels = labels
        
        # Forward pass
        outputs = model(values)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
    train_loss.append(loss.item())
    print ('Epoch [{}/{}], Step [{}/{}], trainLoss: {:.4f}' 
            .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    for values, labels in val_loader:
        outputs = model(values)
        valloss = criterion(outputs, labels)
        val_loss.append(valloss.item())
        print ('Epoch [{}/{}], valLoss: {:.4f}' 
        .format(epoch+1, num_epochs, valloss.item()))



def plot_curve(data1, data2):
    fig = plt.figure()
    plt.plot(range(len(data1)), data1, color='blue', label = "train_loss")
    plt.plot(range(len(data2)), data2, color='yellow', label = "val_loss")
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title("loss curve")
    plt.show()

#plot_curve(train_loss, val_loss)

#torch.save(model.state_dict(), "score/ckpt.mdl")
#model.load_state_dict(torch.load('ckpt.mdl'))

with torch.no_grad():
    correct = 0
    total = len(test_set)
    labeltensor = torch.empty(0,1)
    outputtensor = torch.empty(0,1)
    for values, labels in test_loader:
        values = values
        outputs = model(values)
        labeltensor = torch.cat([labeltensor,labels],dim = 0)
        outputtensor = torch.cat([outputtensor,outputs],dim = 0)
        rule = abs(outputs-labels) < 0.5
        count = rule.sum()
        correct += count

    print(100*correct/total)

'''

'''
roc = ROC(pos_label = 0)
fpr, tpr, thresholds = roc(outputtensor, labeltensor)


def plotROC(fpr,tpr):
    plt.plot(tpr,fpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("ROC curve")
    plt.show()
plotROC(fpr,tpr)


prc = PRC()
precision, recall, thresholds = prc(outputtensor, labeltensor)

def plotPRC(recall,precision):
    plt.plot(recall,precision)
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.title("PRC curve")
    plt.show()
plotPRC(recall,precision)


labeltensor = labeltensor.to(torch.int)
#auroc = AUROC()
#print(auroc(outputtensor, labeltensor))
auprc = AUPRC()
print(auprc(outputtensor, labeltensor))
'''