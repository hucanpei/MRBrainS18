import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class fcn_hed(nn.Module):
    def __init__(self,n_classes=9):
        super(fcn_hed, self).__init__()
        self.n_classes = n_classes
        
        self.pre_conv=nn.Sequential(nn.Conv2d(3,3,1),nn.ReLU(inplace=True),)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)

        self.pool=nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.fcn_conv1_16=nn.Conv2d(64,  64, 3, padding=1)
        self.fcn_conv2_16=nn.Conv2d(128, 64, 3, padding=1)
        self.fcn_conv3_16=nn.Conv2d(256, 64, 3, padding=1)
        self.fcn_conv4_16=nn.Conv2d(512, 64, 3, padding=1)
        self.fcn_conv5_16=nn.Conv2d(512, 64, 3, padding=1)

        self.fcn_up_conv2_16 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.fcn_up_conv3_16 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=4)
        self.fcn_up_conv4_16 = nn.ConvTranspose2d(64, 64, kernel_size=8, stride=8)
        self.fcn_up_conv5_16 = nn.ConvTranspose2d(64, 64, kernel_size=16, stride=16)

        self.fcn_score=nn.Sequential(
            nn.Conv2d(4*64,self.n_classes,1),
            #nn.Dropout(0.5),
            )
        
        self.hed_conv1_1=nn.Conv2d(64,  1, 1)
        self.hed_conv2_1=nn.Conv2d(128, 1, 1)
        self.hed_conv3_1=nn.Conv2d(256, 1, 1)
        self.hed_conv4_1=nn.Conv2d(512, 1, 1)
        self.hed_conv5_1=nn.Conv2d(512, 1, 1)

        self.hed_up_conv2 = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        self.hed_up_conv3 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=4)
        self.hed_up_conv4 = nn.ConvTranspose2d(1, 1, kernel_size=8, stride=8)
        self.hed_up_conv5 = nn.ConvTranspose2d(1, 1, kernel_size=16, stride=16)

        self.hed_score=nn.Sequential(
            nn.Conv2d(4,1,1),
            #nn.Dropout(0.5),
            )



    def forward(self, x):
        #x=self.pre_conv(x)
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(self.pool(conv1))
        conv3 = self.conv_block3(self.pool(conv2))
        conv4 = self.conv_block4(self.pool(conv3))
        #conv5 = self.conv_block5(self.pool(conv4))
        
        fcn_conv1_16=self.fcn_conv1_16(conv1)
        fcn_up_conv2_16=self.fcn_up_conv2_16(self.fcn_conv2_16(conv2))
        fcn_up_conv3_16=self.fcn_up_conv3_16(self.fcn_conv3_16(conv3))
        fcn_up_conv4_16=self.fcn_up_conv4_16(self.fcn_conv4_16(conv4))
        #fcn_up_conv5_16=self.fcn_up_conv5_16(self.fcn_conv5_16(conv5))

        fcn_concat_1_to_4=torch.cat([fcn_conv1_16,fcn_up_conv2_16,fcn_up_conv3_16,fcn_up_conv4_16], 1)
        fcn_score=self.fcn_score(fcn_concat_1_to_4)
        
        hed_conv1_1=self.hed_conv1_1(conv1)
        hed_up_conv2=self.hed_up_conv2(self.hed_conv2_1(conv2))
        hed_up_conv3=self.hed_up_conv3(self.hed_conv3_1(conv3))
        hed_up_conv4=self.hed_up_conv4(self.hed_conv4_1(conv4))
        #hed_up_conv5=self.hed_up_conv5(self.hed_conv5_1(conv5))

        hed_concat_1_to_4=torch.cat([hed_conv1_1,hed_up_conv2,hed_up_conv3,hed_up_conv4], 1)
        hed_score=self.hed_score(hed_concat_1_to_4)
        
        return fcn_score,F.sigmoid(hed_score),F.sigmoid(hed_conv1_1),F.sigmoid(hed_up_conv2),F.sigmoid(hed_up_conv3),F.sigmoid(hed_up_conv4)

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
