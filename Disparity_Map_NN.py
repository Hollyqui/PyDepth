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
import random

import matplotlib.pyplot as plt
import numpy as np

def imageBatch(nb_image):
    imgBatchR_T = torch.randint(0,255,(nb_image,3,9,9))
    imgBatchL_T = imgBatchR_T
    labelT = torch.ones(nb_image,1)

    imgBatchR_F = torch.randint(0,255,(nb_image,3,9,9)) #
    imgBatchL_F = torch.randint(0,255,(nb_image,3,9,9))
    labelF = torch.zeros(nb_image,1)

    finalR = torch.cat((imgBatchR_T,imgBatchR_F))
    finalL = torch.cat((imgBatchL_T,imgBatchL_F))
    finalLabel = torch.cat((labelT,labelF))

    return finalR, finalL, finalLabel

def firstStageCNN():
    return nn.Sequential(nn.Linear(3*9*9, 50), #L1
                         nn.ReLU(inplace=True),
                         
                         nn.Linear(50, 50), #L2
                         nn.ReLU(inplace=True))

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = firstStageCNN()
        
        self.cnn2 = firstStageCNN()

        
        self.fc = nn.Sequential(nn.Linear(100, 75), #L3
                                nn.ReLU(inplace=True),
                                
                                nn.Linear(75, 50), #L4
                                nn.ReLU(inplace=True),
                            
                                nn.Linear(50, 25), #L5
                                nn.ReLU(inplace=True),
                                
                                nn.Linear(25, 1)) #L6
    
    def forward(self, input1, input2):
        
        output1 = self.cnn1(input1.float().view(-1,3*9*9))
        output2 = self.cnn2(input2.float().view(-1,3*9*9))


        combined = torch.cat((output1.view(output1.size(0), -1),
                              output2.view(output2.size(0), -1)), dim=1)

        combined = torch.unsqueeze(combined,2)
        combined = torch.unsqueeze(combined,3)
        combined = combined.view(-1,100)
        
        out = self.fc(combined)
        
        return out

def train(net, finalR, finalL, finalLabel, EPOCHS, BATCH_SIZE):
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    loss_function = nn.MSELoss()
    dataset = utils.TensorDataset(finalR, finalL, finalLabel)
    train_dataloader = DataLoader(dataset, shuffle=True, num_workers=0, batch_size=BATCH_SIZE)
    net.zero_grad()

    print("train function was executed")
    COUNTER = 0
    for epoch in range(EPOCHS):
        for i, data in enumerate(train_dataloader):

            img1, img2, label = data
            optimizer.zero_grad() # reset gradient
            outputs = net(img1, img2)
            loss = loss_function(outputs, label)
            loss.backward()
            optimizer.step()
        #Print out images and epoch numbers 
        print("Epoch number: ", COUNTER)
        COUNTER += 1
        print("Loss:", loss)
    return net

def main():
	net = SiameseNetwork()
	NumberIMG = 500
	EPOCHS = 50

	finalR, finalL, finalLabel = imageBatch(NumberIMG)

	final = train(net,finalR, finalL, finalLabel,EPOCHS,NumberIMG)

if __name__ == '__main__':
    main()
