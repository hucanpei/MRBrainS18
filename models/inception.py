import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

class incep_unit_135(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(incep_unit_135,self).__init__()
        self.ch_in=ch_in
        self.ch_out=ch_out
        self.conv1=nn.Sequential(nn.Conv2d(self.ch_in,self.ch_out,1,padding=0),nn.ReLU(inplace=True),)
        self.conv3=nn.Sequential(nn.Conv2d(self.ch_in,self.ch_out,3,padding=1),nn.ReLU(inplace=True),)
    def forward(self,x):
        conv1=self.conv1(x)
        conv3=self.conv3(x)
        conv5=self.conv3(self.conv3(x))
        return torch.cat([conv1,conv3,conv5],1)

class incep_unit_1357(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(incep_unit_1357,self).__init__()
        self.ch_in=ch_in
        self.ch_out=ch_out
        self.conv1=nn.Sequential(nn.Conv2d(self.ch_in,self.ch_out,1,padding=0),nn.ReLU(inplace=True),)
        self.conv3=nn.Sequential(nn.Conv2d(self.ch_in,self.ch_out,3,padding=1),nn.ReLU(inplace=True),)
    def forward(self,x):
        conv1=self.conv1(x)
        conv3=self.conv3(x)
        conv5=self.conv3(self.conv3(x))
        conv7=self.conv3(self.conv3(self.conv3(x)))
        return torch.cat([conv1,conv3,conv5,conv7],1)

class incep_FCN(nn.Module):
    def __init__(self,n_classes):
        super(incep_FCN,self).__init__()
        self.n_classes=n_classes

        self.stage1=nn.Sequential(nn.Conv2d(3,64,1),incep_unit_135(64,64),)
        #self.stage1=nn.Sequential(incep_unit_135(3,64),incep_unit_135(3*64,64),)
        self.stage2=nn.Sequential(incep_unit_135(3*64,128),incep_unit_135(3*128,128),)
        self.stage3=nn.Sequential(incep_unit_1357(3*128,256),incep_unit_1357(4*256,256),)
        self.stage4=nn.Sequential(incep_unit_1357(4*256,512),incep_unit_1357(4*512,512),)

        self.pool=nn.MaxPool2d(2,stride=2,ceil_mode=True)
        
        self.feature1=nn.Conv2d(3*64,64,3,padding=1)
        self.feature2=nn.Conv2d(3*128,64,3,padding=1)
        self.feature3=nn.Conv2d(4*256,64,3,padding=1)
        self.feature4=nn.Conv2d(4*512,64,3,padding=1)

        self.deconv2=nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.deconv3=nn.ConvTranspose2d(64,64,kernel_size=4,stride=4)
        self.deconv4=nn.ConvTranspose2d(64,64,kernel_size=8,stride=8)

        self.score=nn.Sequential(nn.Conv2d(4*64,self.n_classes,1),)

    def forward(self,x):
        stage1=self.stage1(x)
        stage2=self.stage2(self.pool(stage1))
        stage3=self.stage3(self.pool(stage2))
        stage4=self.stage4(self.pool(stage3))

        feature1=self.feature1(stage1)
        feature2=self.feature2(stage2)
        feature3=self.feature3(stage3)
        feature4=self.feature4(stage4)

        deconv2=self.deconv2(feature2)
        deconv3=self.deconv3(feature3)
        deconv4=self.deconv4(feature4)

        cat=torch.cat([feature1,deconv2,deconv3,deconv4],1)
        score=self.score(cat)
        return score

if __name__ =='__main__':
    x=torch.Tensor(4,3,240,240)
    x=Variable(x.cuda())
    model=incep_FCN(n_classes=11)
    model.cuda()
    y=model(x)
    print(y.shape)







