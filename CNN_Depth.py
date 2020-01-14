import os
import torch
import torchvision
import torch.utils.data as utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchsummary import summary
import timeit
import random

import matplotlib.pyplot as plt
import numpy as np


def imageBatch(nb_image):
    imgBatch = torch.rand(nb_image, 3, 640, 480)
    return imgBatch


def firstStageCNN():
    return nn.Sequential(nn.Conv2d(3, 10, kernel_size=3),  # optional: add stride
                         nn.ReLU(inplace=True),
                         nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1),

                         nn.Conv2d(10, 17, kernel_size=3),
                         nn.ReLU(inplace=True),
                         nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1),
                         nn.MaxPool2d(kernel_size=3),  # optional: add stride
                         nn.ReLU(inplace=True),

                         nn.Conv2d(17, 25, kernel_size=3),  # optional: add stride
                         nn.ReLU(inplace=True),
                         nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1),

                         nn.MaxPool2d(kernel_size=3),  # optional: add stride
                         nn.ReLU(inplace=True))


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = firstStageCNN()

        self.cnn2 = firstStageCNN()

        self.fc = nn.Sequential(nn.Conv2d(182000, 2, kernel_size=1),
                                nn.ReLU(inplace=True),

                                nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.ReLU(inplace=True),

                                nn.Conv2d(2, 36, kernel_size=1),
                                nn.ReLU(inplace=True),

                                nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.ReLU(inplace=True),

                                nn.Conv2d(36, 80, kernel_size=1),
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

        combined = torch.unsqueeze(combined, 2)
        combined = torch.unsqueeze(combined, 3)
        out = self.fc(combined)

        return output1


# NumberIMG is the amount of images in one EPOCH
# training_DATA_LEFT etc is the tensor array with the whole dataset
# for the moment it is training_DATA_LEFT = imageBatch(NumberIMG)
def train(net, EPOCHS, NumberIMG, BATCH_SIZE, training_DATA_LEFT, training_DATA_RIGHT, depthMaps):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    net.zero_grad()

    training_DATA_LEFT = imageBatch(NumberIMG)
    training_DATA_RIGHT = imageBatch(NumberIMG)
    depthMaps = depthMaps(NumberIMG)

    print("train function was executed")
    for epoch in range(EPOCHS):
        COUNTER = 0
        listOfIndexes = random.shuffle(list(range(NumberIMG)))

        for batch in range(NumberIMG / BATCH_SIZE):
            leftList = []
            rightList = []
            depthList = []

            for img in range(BATCH_SIZE):
                leftList.append(training_DATA_LEFT[listOfIndexes[COUNTER]])
                rightList.append(training_DATA_RIGHT[listOfIndexes[COUNTER]])
                depthList.append(depthMaps[listOfIndexes[COUNTER]])
                COUNTER += 1

            leftBatch = torch.stack(tuple(leftList), 1)
            rightBatch = torch.stack(tuple(rightList), 1)
            depthMapBatch = torch.stack(tuple(depthList), 1)

            optimizer.zero_grad()

            outputs = net(leftBatch, rightBatch)

            loss = loss_function(outputs, depthMapBatch)

            loss.backward()
            optimizer.step()

            # Printing progression
            if COUNTER % 10 == 0:
                print("Epoch number")


        return net


def main():
    net = SiameseNetwork()

    # This will import the real dataset in tensor arrays once the data is available
    training_DATA_LEFT = imageBatch(NumberIMG)
    training_DATA_RIGHT = imageBatch(NumberIMG)
    depthMaps = depthMaps(NumberIMG)

    final = train(net, EPOCHS=4, NumberIMG=10, BATCH_SIZE=5, training_DATA_LEFT, training_DATA_RIGHT, depthMaps)


if __name__ == '__main__':
    main()
