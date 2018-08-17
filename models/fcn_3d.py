import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class fcn_3d(nn.Module):
    def __init__(self,n_classes=11):
        super(fcn_3d,self).__init__()
        self.n_classes=n_classes
        self.conv_block1=nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1),nn.ReLU(inplace=True),)
            #nn.Conv3d(64, 64, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(64, 128, 3, padding=1),nn.ReLU(inplace=True),)
            #nn.Conv3d(128, 128, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1),nn.ReLU(inplace=True),)
            #nn.Conv3d(256, 256, 3, padding=1),nn.ReLU(inplace=True),
            #nn.Conv3d(256, 256, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block4 = nn.Sequential(
            nn.Conv3d(256, 512, 3, padding=1),nn.ReLU(inplace=True),)
            #nn.Conv3d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            #nn.Conv3d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)
        self.pool=nn.MaxPool3d(2, stride=2, ceil_mode=True)
        self.conv1_16=nn.Conv3d(64,  64, 3, padding=1)
        self.conv2_16=nn.Conv3d(128, 64, 3, padding=1)
        self.conv3_16=nn.Conv3d(256, 64, 3, padding=1)
        self.conv4_16=nn.Conv3d(512, 64, 3, padding=1)
        self.up_conv2_16 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.up_conv3_16 = nn.ConvTranspose3d(64, 64, kernel_size=4, stride=4)
        self.up_conv4_16 = nn.ConvTranspose3d(64, 64, kernel_size=8, stride=8)
        self.score=nn.Sequential(
            nn.Conv3d(4*64,self.n_classes,1),
            #nn.Dropout(0.5),
            )
    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(self.pool(conv1))
        conv3 = self.conv_block3(self.pool(conv2))
        conv4 = self.conv_block4(self.pool(conv3))
        
        conv1_16=self.conv1_16(conv1)
        up_conv2_16=self.up_conv2_16(self.conv2_16(conv2))
        up_conv3_16=self.up_conv3_16(self.conv3_16(conv3))
        up_conv4_16=self.up_conv4_16(self.conv4_16(conv4))

        concat_1_to_4=torch.cat([conv1_16,up_conv2_16,up_conv3_16,up_conv4_16], 1)
        score=self.score(concat_1_to_4)
        return score
    
