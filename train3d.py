import os
import nibabel as nib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils import data

def readVol(volpath):
    return nib.load(volpath).get_data()

def to_uint8(vol):
    vol=vol.astype(np.float)
    vol[vol<0]=0
    return ((vol-vol.min())*255.0/vol.max()).astype(np.uint8)

class MR18_3d(data.Dataset):
    def __init__(self,root='../../data',is_val=False,val_num=1):
        self.root=root
        self.is_val=is_val
        self.val_num=val_num
        self.n_classes=11
        self.T1path=[self.root+'training/'+name+'/pre/reg_T1.nii.gz' for name in ['1','4','5','7','14','070','148']]
        self.lblpath=[self.root+'training/'+name+'/segm.nii.gz' for name in ['1','4','5','7','14','070','148']]
        self.val_T1path=self.T1path[self.val_num-1]
        self.val_lblpath=self.lblpath[self.val_num-1]
        self.train_T1path=[temp for temp in self.T1path if temp not in [self.val_T1path]]
        self.train_lblpath=[temp for temp in self.lblpath if temp not in [self.val_lblpath]]
        if self.is_val==False:
            print('training data')
            T1_nii=[to_uint8(readVol(path)) for path in self.train_T1path]
            lbl_nii=[readVol(path) for path in self.train_lblpath]
            print('transforming')
            for sample_index in range(6):
                T1_nii[sample_index],lbl_nii[sample_index]= \
                        self.transform(T1_nii[sample_index],lbl_nii[sample_index])
        else:
            print('validating data')
            T1_nii=to_uint8(readVol(self.val_T1path))
            lbl_nii=readVol(self.val_lblpath)
            T1_nii,lbl_nii=self.transform(T1_nii,lbl_nii)
        self.T1_nii=T1_nii
        self.lbl_nii=lbl_nii
    def __len__(self):
        return (self.is_val)and(1)or(6)
    def __getitem__(self,index):
        if not self.is_val:
            return self.T1_nii[index],self.lbl_nii[index]
        else:
            return self.T1_nii,self.lbl_nii
    def transform(self,T1,lbl):
        T1=torch.from_numpy((T1.transpose(2,0,1).astype(np.float)-0.0)/255.0).unsqueeze(0).float()
        lbl=torch.from_numpy(lbl.transpose(2,0,1)).unsqueeze(0).long()
        return T1,lbl

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


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, d, h, w = input.size()
    nt, ct, dt, ht, wt = target.size()
    '''
    # Handle inconsistent size between input and target
    if h > ht and w > wt: # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode='nearest')
        target = target.sequeeze(1)
    elif h < ht and w < wt: # upsample images
        input = F.upsample(input, size=(ht, wt), mode='bilinear')
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")
    '''
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    log_p = log_p[target.contiguous().view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum().float()
    return loss

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train():
    os.environ["CUDA_VISIBLE_DEVICES"]=str(1)
    #torch.manual_seed(1337)
    data_path='/home/canpi/MRBrainS18/data/'
    t_loader=MR18_3d(root=data_path,is_val=False,val_num=1)
    trainloader=data.DataLoader(t_loader,batch_size=1,num_workers=4,shuffle=True)
    v_loader=MR18_3d(root=data_path,is_val=True,val_num=1)
    valloader=data.DataLoader(v_loader,batch_size=1,num_workers=4,shuffle=False)
    n_classes=t_loader.n_classes
    model=fcn_3d(n_classes)
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=5e-4)
    loss_ce = cross_entropy2d
    for epoch in range(100):
        for i,(T1,lbl) in enumerate(trainloader):
            T1,lbl=Variable(T1.cuda()),Variable(lbl.cuda())
            optimizer.zero_grad()
            out=model(T1)
            loss=loss_ce(input=out,target=lbl)
            loss.backward()
            optimizer.step()
        print(loss.item())


if __name__ == '__main__':
    train()
