import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class coord_conv(nn.Module):
    def __init__(self,ch_in,ch_out,kern=3,pad=1):
        super(coord_conv,self).__init__()
        self.ch_in=ch_in
        self.ch_out=ch_out
        self.kern=kern
        self.conv=nn.Conv2d(ch_in,ch_out-2,kern,padding=pad)
    def forward(self,x,coord):
        return self.conv(torch.cat([x,coord],1))

class coord_fcn(nn.Module):
    def __init__(self,n_classes=9):
        super(coord_fcn, self).__init__()
        self.n_classes = n_classes
        self.relu=nn.ReLU(inplace=True)

        #self.conv1_1=coord_conv(5  ,64 ,3)
        #self.conv1_2=coord_conv(64 ,64 ,3)
        #self.conv2_1=coord_conv(64 ,128,3)
        #self.conv2_2=coord_conv(128,128,3)
        #self.conv3_1=coord_conv(128,256,3)
        #self.conv3_2=coord_conv(256,256,3)
        #self.conv3_3=coord_conv(256,256,3)
        #self.conv4_1=coord_conv(256,512,3)
        #self.conv4_2=coord_conv(512,512,3)
        #self.conv4_3=coord_conv(512,512,3)
        
        self.conv1_1=nn.Conv2d(3  ,64 ,3,padding=1)
        self.conv1_2=nn.Conv2d(64 ,64 ,3,padding=1)
        self.conv2_1=nn.Conv2d(64 ,128,3,padding=1)
        self.conv2_2=nn.Conv2d(128,128,3,padding=1)
        self.conv3_1=nn.Conv2d(128,256,3,padding=1)
        self.conv3_2=nn.Conv2d(256,256,3,padding=1)
        self.conv3_3=nn.Conv2d(256,256,3,padding=1)
        self.conv4_1=nn.Conv2d(256,512,3,padding=1)
        self.conv4_2=nn.Conv2d(512,512,3,padding=1)
        self.conv4_3=nn.Conv2d(512,512,3,padding=1)


        self.conv1=coord_conv(64+2 ,64+2,3)
        self.conv2=coord_conv(128+2,64+2,3)
        self.conv3=coord_conv(256+2,64+2,3)
        self.conv4=coord_conv(512+2,64+2,3)
        #self.conv1=nn.Conv2d(64-2,64,3,padding=1)
        #self.conv2=nn.Conv2d(128-2,64,3,padding=1)
        #self.conv3=nn.Conv2d(256-2,64,3,padding=1)
        #self.conv4=nn.Conv2d(512-2,64,3,padding=1)

        self.score=coord_conv(4*64+2,self.n_classes+2,1,0)
        #self.score=nn.Conv2d(4*64,self.n_classes,1)

    def forward(self, x,coord):
        conv1_1=self.relu(self.conv1_1(x))
        conv1_2=self.relu(self.conv1_2(conv1_1))
        conv2_1=self.relu(self.conv2_1(conv1_2))
        conv2_2=self.relu(self.conv2_2(conv2_1))
        conv3_1=self.relu(self.conv3_1(conv2_2))
        conv3_2=self.relu(self.conv3_2(conv3_1))
        conv3_3=self.relu(self.conv3_3(conv3_2))
        conv4_1=self.relu(self.conv4_1(conv3_3))
        conv4_2=self.relu(self.conv4_2(conv4_1))
        conv4_3=self.relu(self.conv4_3(conv4_2))

        #conv1=self.conv1(conv1_2,coord)
        #conv2=self.conv2(conv2_2,coord)
        #conv3=self.conv3(conv3_3,coord)
        #conv4=self.conv4(conv4_3,coord)

        #concat=torch.cat([conv1,conv2,conv3,conv4],1)
        #score=self.score(concat,coord)

        return conv4_3
