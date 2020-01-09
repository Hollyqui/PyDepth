#Import libraries
import torchvision
import torch.utils.data as utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import time
import copy
from torch.optim import lr_scheduler
import os
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd

#Global Variables
EPOCHS = 3


#main code
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(

        nn.Conv2d(1,96, kernel_size=3), #optional: add stride
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=1),

        nn.Conv2d(96,256, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=1),

        nn.MaxPool2d(kernel_size=3), #optional: add stride
        nn.ReLU(inplace=True),

        nn.Conv2d(256,384, kernel_size=3), #optional: add stride
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=1),

        nn.MaxPool2d(kernel_size=3), #optional: add stride
        nn.ReLU(inplace=True),
        )

        self.cnn2 = nn.Sequential (

        nn.Conv2d(1,96, kernel_size=3), #optional: add stride
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=1),

        nn.Conv2d(96,256, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=1),

        nn.MaxPool2d(kernel_size=3), #optional: add stride
        nn.ReLU(inplace=True),

        nn.Conv2d(256,384, kernel_size=3), #optional: add stride
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=1),

        nn.MaxPool2d(kernel_size=3), #optional: add stride
        nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
        nn.Conv2d(1024,384,kernel_size=3),
        nn.ReLU(inplace=True),

        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReLU(inplace=True),

        nn.Conv2d(384,256,kernel_size=3),
        nn.ReLU(inplace=True),

        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReLU(inplace=True),

        nn.Conv2d(384,256,kernel_size=3),
        nn.ReLU(inplace=True),
        # nn.Linear(30976, 1024), #input and output features
        # nn.ReLU(inplace=True),
        #
        # nn.Linear(1024, 128), #input and output features
        # nn.ReLU(inplace=True),
        #
        # nn.Linear(128,2)

        )

    def forward(self, input1, input2):
      cnn1 = self.cnn1(input1)
      cnn2 = self.cnn2(input2)
      # now we can reshape `c` and `f` to 2D and concat them
      combined = torch.cat((cnn1.view(cnn1.size(0), -1),
                            cnn2.view(cnn2.size(0), -1)), dim=1)
      out = self.fc2(combined)
      return out

# #concatenate the layers
# inputForLastNN = torch.cat((outputNN1, outputNN2), 1),
# Add the functions after the two outputs are concatinated and then move on to the loss function!



left_data = np.load('C:/Users/szymo/Documents/left_images_numpy.npy')
right_data = np.load('C:/Users/szymo/Documents/right_images_numpy.npy')
depth_map = np.load('C:/Users/szymo/Documents/depthmaps_numpy.npy')
#Train the Model


if torch.cuda.is_available(): # Check whether you have GPU is loaded or not
    print('Yes')

# Declare Siamese Network
net = SiameseNetwork()
# Decalre Loss Function






# additional functions
def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()



# Training the Network
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(EPOCHS):



# -------------- Appendix ---------------------- Appendix ---------------------- Appendix ----------------- Appendix -----------------

#concatenate nets exemple
class TwoInputsNet(nn.Module):
  def __init__(self):
    super(TwoInputsNet, self).__init__()
    self.conv = nn.Conv2d( ... )  # set up your layer here
    self.fc1 = nn.Linear( ... )  # set up first FC layer
    self.fc2 = nn.Linear( ... )  # set up the other FC layer

  def forward(self, input1, input2):
    c = self.conv(input1)
    f = self.fc1(input2)
    # now we can reshape `c` and `f` to 2D and concat them
    combined = torch.cat((c.view(c.size(0), -1),
                          f.view(f.size(0), -1)), dim=1)
    out = self.fc2(combined)
    return out


#Keras exemple code
def disparity_cnn_model(input_shape):
    shape=(None, input_shape[1], input_shape[2], input_shape[3])
    left = Input(batch_shape=shape)
    right = Input(batch_shape=shape)

    left_1 = Conv2D(filters=32, kernel_size=3, padding='same')(left)
    left_1_pool = MaxPooling2D(2)(left_1)
    left_1_activate = Activation('relu')(left_1_pool)

    left_2 = Conv2D(filters=62, kernel_size=3, padding='same')(left_1_activate)
    left_2_pool = MaxPooling2D(2)(left_2)
    left_2_activate = Activation('relu')(left_2_pool)

    left_3 = Conv2D(filters=92, kernel_size=3, padding='same')(left_2_activate)
    left_3_activate = Activation('relu')(left_3)

    right_1 = Conv2D(filters=32, kernel_size=3, padding='same')(right)
    right_1_pool = MaxPooling2D(2)(right_1)
    right_1_activate = Activation('relu')(right_1_pool)

    right_2 = Conv2D(filters=62, kernel_size=3, padding='same')(right_1_activate)
    right_2_pool = MaxPooling2D(2)(right_2)
    right_2_activate = Activation('relu')(right_2_pool)

    right_3 = Conv2D(filters=92, kernel_size=3, padding='same')(right_2_activate)
    right_3_activate = Activation('relu')(right_3)

    merge = concatenate([left_3_activate, right_3_activate])

    merge_1 = Conv2D(filters=62, kernel_size=3, padding='same')(merge)
    merge_1_up = UpSampling2D(2)(merge_1)
    merge_1_activate = Activation('relu')(merge_1_up)

    merge_2 = Conv2D(filters=22, kernel_size=3, padding='same')(merge_1_activate)
    merge_2_up = UpSampling2D(2)(merge_2)
    merge_2_activate = Activation('relu')(merge_2_up)

    merge_3 = Conv2D(filters=1, kernel_size=2, padding='same')(merge_2_activate)
    merge_3_activate = Activation('relu')(merge_3)

    model = Model([left, right], merge_3_activate)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    return model
