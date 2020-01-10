import os
import torch
import torchvision
import torch.utils.data as utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import timeit

import matplotlib.pyplot as plt
import numpy as np

def imageBatch(nb_image):
    imgBatch = torch.rand(nb_image,3,640,480)
    return imgBatch

def firstStageCNN():
    return nn.Sequential(nn.Conv2d(3,10, kernel_size=3), #optional: add stride
                                  nn.ReLU(inplace=True),
                                  nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=1),
                                  
                                  nn.Conv2d(10,17, kernel_size=3),
                                  nn.ReLU(inplace=True),
                                  nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=1),
                                  nn.MaxPool2d(kernel_size=3), #optional: add stride
                                  nn.ReLU(inplace=True),
                                  
                                  nn.Conv2d(17,25, kernel_size=3), #optional: add stride
                                  nn.ReLU(inplace=True),
                                  nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=1),
                                  
                                  nn.MaxPool2d(kernel_size=3), #optional: add stride
                                  nn.ReLU(inplace=True))

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = firstStageCNN()
        
        self.cnn2 = firstStageCNN()

        self.fc = nn.Sequential(nn.Conv2d(182000,2,kernel_size=1),
                                nn.ReLU(inplace=True),
                                
                                nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.ReLU(inplace=True),
                                
                                nn.Conv2d(2,36,kernel_size=1),
                                nn.ReLU(inplace=True),
                                
                                nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.ReLU(inplace=True),
                                
                                nn.Conv2d(36,80,kernel_size=1),
                                nn.ReLU(inplace=True))
                                
                                # nn.Linear(30976, 1024), #input and output features
                                # nn.ReLU(inplace=True),
                                #
                                # nn.Linear(1024, 128), #input and output features
                                # nn.ReLU(inplace=True),
                                #
                                # nn.Linear(128,2)
    
    def forward(self, input1, input2):
        output1 = self.cnn1(input1)
        output2 = self.cnn2(input2)

        combined = torch.cat((output1.view(output1.size(0), -1),
                              output2.view(output2.size(0), -1)), dim=1)

        combined = torch.unsqueeze(combined,2)
        combined = torch.unsqueeze(combined,3)
        out = self.fc(combined)

        return output1


net = SiameseNetwork()

X_l = imageBatch(10)
X_r = imageBatch(10)

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()
net.zero_grad()

print(X_l.shape)

start = timeit.timeit()
outputs = net(X_l, X_r)
end = timeit.timeit()
print("Runtime:", end-start)

loss = loss_function(outputs, outputs)
loss.backward()
optimizer.step()

print(loss)







#TRAINING

#Global variables
EPOCHS = 4
NumberIMG = 10
BATCH_SIZE = 5

#Some important functions
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

net.zero_grad()

training_DATA_LEFT = imageBatch(NumberIMG)
training_DATA_RIGHT = imageBatch(NumberIMG)
depthMaps = depthMaps(NumberIMG)



def train():
  print("train function was executed")
  for epoch in range(EPOCHS):
    COUNTER = 0
    listOfIndexes = suffle(list(range(NumberIMG)))

    for batch in range(NumberIMG/BATCH_SIZE):
      leftList = []
      rightList = []
      depthList = []

      for img in range(BATCH_SIZE):
        leftList.append(training_DATA_LEFT[listOfIndexes[COUNTER]])
        rightList.append(training_DATA_RIGHT[listOfIndexes[COUNTER]])
        depthList.append(depthMaps[listOfIndexes[COUNTER]])
        COUNTER+=1
      
      leftBatch = torch.stack((leftList[0],leftList[1],leftList[2],leftList[3],leftList[4]), 1) 
      rightBatch = torch.stack((rightList[0],rightList[1],rightList[2],rightList[3],rightList[4]), 1)
      depthMapBatch = torch.stack((depthList[0], depthList[1], depthList[2], depthList[3], depthList[4]), 1)

      optimizer.zero_grad()
      
      outputs = net(leftBatch, rightBatch)

      loss = loss_function(outputs, depthMapBatch)
      
      loss.backward()
      optimizer.step()

      #Printing progression
      if counter %10 == 0:
        print("Epoch number")

  return net
