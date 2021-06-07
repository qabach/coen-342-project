'''
Santa Clara University 
COEN 342 - Deep Learning
Spring 2021

Team Project 

Author: Quan Bach

This file is the implementation of the Deconvolution Neural Network. 
This network is used to train on the OASIS dataset to perform depth estimation. 
  
'''

import torch
import torch.nn as nn
from torch.autograd import Variable

class DeconvoNetwork(nn.Module):

    def __init__(self):
        super(DeconvoNetwork,self).__init__()
        #Convolution 1
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64, kernel_size=4,stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight) #Xaviers Initialisation
        self.activation1= nn.ReLU()

        #Max Pool 1
        self.maxpool1= nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.activation2 = nn.ReLU()

        #Max Pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2,return_indices=True)

        #Convolution 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.activation3 = nn.ReLU()

        #De Convolution 1
        self.deconv1=nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3)
        nn.init.xavier_uniform_(self.deconv1.weight)
        self.activation4=nn.ReLU()

        #Max UnPool 1
        self.maxunpool1=nn.MaxUnpool2d(kernel_size=2)

        #De Convolution 2
        self.deconv2=nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=5)
        nn.init.xavier_uniform_(self.deconv2.weight)
        self.activation5=nn.ReLU()

        #Max UnPool 2
        self.maxunpool2=nn.MaxUnpool2d(kernel_size=2)

        #DeConvolution 3
        self.deconv3=nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=4)
        nn.init.xavier_uniform_(self.deconv3.weight)
        self.activation6=nn.ReLU()

    def forward(self,x):
        out=self.conv1(x)
        out=self.activation1(out)
        size1 = out.size()
        out,indices1=self.maxpool1(out)
        out=self.conv2(out)
        out=self.activation2(out)
        size2 = out.size()
        out,indices2=self.maxpool2(out)
        out=self.conv3(out)
        out=self.activation3(out)

        out=self.deconv1(out)
        out=self.actiation4(out)
        out=self.maxunpool1(out,indices2,size2)
        out=self.deconv2(out)
        out=self.actiation5(out)
        out=self.maxunpool2(out,indices1,size1)
        out=self.deconv3(out)
        out=self.activation6(out)
        return(out)