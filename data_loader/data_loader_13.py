import os
import torch
import numpy as np
import math
import random
import cv2 as cv
import nibabel as nib
import torch
from torch.utils import data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class MR13loader_CV(data.Dataset):
    def __init__(self,root='../../data13/',val_num=5,is_val=False,
                 is_transform=False,is_rotate=False,is_crop=False,is_histeq=False,forest=5):
        self.root=root
        self.val_num=val_num
        self.is_val=is_val
        self.is_transform=is_transform
        self.is_rotate=is_rotate
        self.is_crop=is_crop
        self.is_histeq=is_histeq
        self.forest=forest
        self.n_classes=9
        self.T1_classes=9
        self.IR_classes=2
        self.T2_classes=2
        # Back: Background
        # GM:   Cortical GM(red), Basal ganglia(green)
        # WM:   WM(yellow), WM lesions(blue)
        # CSF:  CSF(pink), Ventricles(light blue)
        # Back: Cerebellum(white), Brainstem(dark red)
        self.color=np.asarray([[0,0,0],[0,0,255],[0,255,0],[0,255,255],[255,0,0],\
                [255,0,255],[255,255,0],[255,255,255],[0,0,128],[0,128,0],[128,0,0]]).astype(np.uint8)
        # Back , CSF , GM , WM
        self.label_test=[0,2,2,3,3,1,1,0,0]
        # nii paths
        self.T1path=[self.root+'TrainingData/'+str(num+1)+'/T1.nii' for num in range(5)]
        self.IRpath=[self.root+'TrainingData/'+str(num+1)+'/T1_IR.nii' for num in range(5)]
        self.T2path=[self.root+'TrainingData/'+str(num+1)+'/T2_FLAIR.nii' for num in range(5)]
        self.lblpath=[self.root+'TrainingData/'+str(num+1)+'/LabelsForTraining.nii' for num in range(5)]

        # val path
        self.val_T1path=self.T1path[self.val_num-1]
        self.val_IRpath=self.IRpath[self.val_num-1]
        self.val_T2path=self.T2path[self.val_num-1]
        self.val_lblpath=self.lblpath[self.val_num-1]
        # train path
        self.train_T1path=[temp for temp in self.T1path if temp not in [self.val_T1path]]
        self.train_IRpath=[temp for temp in self.IRpath if temp not in [self.val_IRpath]]
        self.train_T2path=[temp for temp in self.T2path if temp not in [self.val_T2path]]
        self.train_lblpath=[temp for temp in self.lblpath if temp not in [self.val_lblpath]]
        # train nii
        self.trainT1nii=np.array([self.to_uint8(self.readVol(x)) for i,x in enumerate(self.train_T1path)])
        self.trainIRnii=np.array([self.to_uint8(self.readVol(x)) for i,x in enumerate(self.train_IRpath)])
        self.trainT2nii=np.array([self.to_uint8(self.readVol(x)) for i,x in enumerate(self.train_T2path)])
        self.trainlblnii=np.array([self.readVol(x) for i,x in enumerate(self.train_lblpath)])
        # val nii
        self.valT1nii=self.to_uint8(self.readVol(self.val_T1path))
        self.valIRnii=self.IR_to_uint8(self.readVol(self.val_IRpath))
        self.valT2nii=self.to_uint8(self.readVol(self.val_T2path))
        self.vallblnii=self.readVol(self.val_lblpath)
        # hist equalize
        self.histeq()

        self.T1mean=100.0
        self.IRmean=0.0
        self.T2mean=0.0

    def __len__(self):
        return (self.is_val)and(48)or(4*48)
    def __getitem__(self,index):
        # get train or validation data
        if self.is_val==False:
            index_list=range(4)
            set_index=int(index/48)
            img_index=int(index%48)
            imgT1=np.array([self.trainT1nii[index_list[set_index]][:,:,x].transpose() for i,x \
                          in enumerate(self.get_stackindex(img_index,self.forest,48))])
            imgIR=np.array([self.trainIRnii[index_list[set_index]][:,:,x].transpose() for i,x \
                          in enumerate(self.get_stackindex(img_index,self.forest,48))])
            imgT2=np.array([self.trainT2nii[index_list[set_index]][:,:,x].transpose() for i,x \
                          in enumerate(self.get_stackindex(img_index,self.forest,48))])
            lbl=np.array([self.trainlblnii[index_list[set_index]][:,:,x].transpose() for i,x \
                          in enumerate(self.get_stackindex(img_index,self.forest,48))])
        else:
            index_list=range(48)
            imgT1=np.array([self.valT1nii[:,:,x].transpose() for i,x \
                          in enumerate(self.get_stackindex(index_list[index],self.forest,48))])
            imgIR=np.array([self.valIRnii[:,:,x].transpose() for i,x \
                          in enumerate(self.get_stackindex(index_list[index],self.forest,48))])
            imgT2=np.array([self.valT2nii[:,:,x].transpose() for i,x \
                          in enumerate(self.get_stackindex(index_list[index],self.forest,48))])
            lbl=np.array([self.vallblnii[:,:,x].transpose() for i,x \
                          in enumerate(self.get_stackindex(index_list[index],self.forest,48))])
        # get rotate
        if self.is_rotate==True:
            angle_list=[-15,-10,-5,0,5,10,15]
            angle=angle_list[random.randint(0,6)]
            #angle=random.uniform(-15,15)
            for i in range(self.forest):
                imgT1[i],imgIR[i],imgT2[i],lbl[i]=self.rotate(imgT1[i],imgIR[i],imgT2[i],lbl[i],angle)
        # get crop
        if self.is_crop==True:
            region,imgT1,imgIR,imgT2,lbl=self.tuple_crop_pad(imgT1,imgIR,imgT2,lbl)
        edge=self.get_edge(lbl,kernel_size=(2,2),iterations=2)
        # transform
        if self.is_transform==True:
            imgT1,imgIR,imgT2,lbl=self.transform(imgT1,imgIR,imgT2,lbl)
            edge=torch.from_numpy(edge*2.0/255.0).float()
        # return
        if self.is_crop==True:
            return region,imgT1,imgIR,imgT2,lbl,edge
        else:
            return imgT1,imgIR,imgT2,lbl,edge
    # read nii
    def readVol(self,volpath):
        return nib.load(volpath).get_data()
    # convert nii to uint8
    def to_uint8(self,vol):
        vol=vol.astype(np.float)
        vol[vol<0]=0
        return ((vol-vol.min())*255.0/vol.max()).astype(np.uint8)
    def IR_to_uint8(self,vol):
        vol=vol.astype(np.float)
        print(vol.min())
        vol[vol<0]=0
        return ((vol-800)*255.0/vol.max()).astype(np.uint8)
    # hist equalize
    def histeq(self):
        if self.is_histeq==True:
            for vol_num in range(4):
                for slice_num in range(48):
                    self.trainT1nii[vol_num,:,:,slice_num]=cv.equalizeHist(self.trainT1nii[vol_num,:,:,slice_num])
                    self.trainIRnii[vol_num,:,:,slice_num]=cv.equalizeHist(self.trainIRnii[vol_num,:,:,slice_num])
                    self.trainT2nii[vol_num,:,:,slice_num]=cv.equalizeHist(self.trainT2nii[vol_num,:,:,slice_num])
            for slice_num in range(48):
                self.valT1nii[:,:,slice_num]=cv.equalizeHist(self.valT1nii[:,:,slice_num])
                self.valIRnii[:,:,slice_num]=cv.equalizeHist(self.valIRnii[:,:,slice_num])
                self.valT2nii[:,:,slice_num]=cv.equalizeHist(self.valT2nii[:,:,slice_num])
    # stack image index
    def get_stackindex(self,index,num,slice_num):
        assert num%2==1, 'num must be odd!'
        list_index=[0]*num
        for i in range(num):
            list_index[i]=(index+(i-int(num/2)))%slice_num
        return list_index
    def rotate(self,imgT1,imgIR,imgT2,lbl,angle):
        raws,cols=imgT1.shape
        M=cv.getRotationMatrix2D(((cols-1)/2.0,(raws-1)/2.0),angle,1)
        imgT1_rotated=cv.warpAffine(imgT1,M,(cols,raws))
        imgIR_rotated=cv.warpAffine(imgIR,M,(cols,raws))
        imgT2_rotated=cv.warpAffine(imgT2,M,(cols,raws))
        lbl_rotated=cv.warpAffine(lbl,M,(cols,raws),flags=cv.INTER_NEAREST)
        return imgT1_rotated,imgIR_rotated,imgT2_rotated,lbl_rotated
    # get crop region for one img
    def calc_crop_region(self,imgT1,thre,pix):
        _,threimg=cv.threshold(imgT1.copy(),thre,255,cv.THRESH_TOZERO)
        pix_index=np.where(threimg>0)
        if not pix_index[0].size == 0:
            y_min,y_max=min(pix_index[0]),max(pix_index[0])
            x_min,x_max=min(pix_index[1]),max(pix_index[1])
        else:
            y_min,y_max=pix,pix
            x_min,x_max=pix,pix
        y_min=(y_min<=pix)and(0)or(y_min)
        y_max=(y_max>=imgT1.shape[0]-1-pix)and(imgT1.shape[0]-1)or(y_max)
        x_min=(x_min<=pix)and(0)or(x_min)
        x_max=(x_max>=imgT1.shape[1]-1-pix)and(imgT1.shape[1]-1)or(x_max)
        return [y_min,y_max,x_min,x_max]
    # pad img to divide by 16
    def calc_ceil_pad(self,x,num):
        return (math.ceil(x/float(num))*num)
    # take tuple crop acording to the max crop region
    def tuple_crop_pad(self,imgT1_tuple,imgIR_tuple,imgT2_tuple,lbl_tuple):
        y_min_list,y_max_list,x_min_list,x_max_list=[],[],[],[]
        for ii in range(self.forest):
            region=self.calc_crop_region(imgT1_tuple[ii],50,5)
            y_min_list.append(region[0])
            y_max_list.append(region[1])
            x_min_list.append(region[2])
            x_max_list.append(region[3])
        # get max crop region
        y_min,y_max,x_min,x_max=min(y_min_list),max(y_max_list),min(x_min_list),max(x_max_list)
        imgT1_tuple_cropped=np.zeros((self.forest,self.calc_ceil_pad(y_max-y_min,16),self.calc_ceil_pad(x_max-x_min,16)),np.uint8)
        imgIR_tuple_cropped=np.zeros((self.forest,self.calc_ceil_pad(y_max-y_min,16),self.calc_ceil_pad(x_max-x_min,16)),np.uint8)
        imgT2_tuple_cropped=np.zeros((self.forest,self.calc_ceil_pad(y_max-y_min,16),self.calc_ceil_pad(x_max-x_min,16)),np.uint8)
        lbl_tuple_cropped=np.zeros((self.forest,self.calc_ceil_pad(y_max-y_min,16),self.calc_ceil_pad(x_max-x_min,16)),np.uint8)
        imgT1_tuple_cropped[:,0:y_max-y_min,0:x_max-x_min]=imgT1_tuple[:,y_min:y_max,x_min:x_max]
        imgIR_tuple_cropped[:,0:y_max-y_min,0:x_max-x_min]=imgIR_tuple[:,y_min:y_max,x_min:x_max]
        imgT2_tuple_cropped[:,0:y_max-y_min,0:x_max-x_min]=imgT2_tuple[:,y_min:y_max,x_min:x_max]
        lbl_tuple_cropped[:,0:y_max-y_min,0:x_max-x_min]=lbl_tuple[:,y_min:y_max,x_min:x_max]
        return np.array([y_min,y_max,x_min,x_max]),imgT1_tuple_cropped,imgIR_tuple_cropped,imgT2_tuple_cropped,lbl_tuple_cropped
    def get_edge(self,lbl,kernel_size=(2,2),iterations=1):
        edge=np.zeros((lbl.shape[0],lbl.shape[1],lbl.shape[2]),np.uint8)
        kernel=np.ones(kernel_size,np.uint8)
        for i in range(len(lbl)):
            edge[i]=cv.Canny(lbl[i].copy(),1,1)
            edge[i]=cv.dilate(edge[i],kernel,iterations=iterations)
        return edge
    def transform(self,imgT1,imgIR,imgT2,lbl):
        imgT1=torch.from_numpy((imgT1.astype(np.float)-self.T1mean)/255.0).float()
        imgIR=torch.from_numpy((imgIR.astype(np.float)-self.IRmean)/255.0).float()
        imgT2=torch.from_numpy((imgT2.astype(np.float)-self.T2mean)/255.0).float()
        lbl=torch.from_numpy(lbl).long()
        return imgT1,imgIR,imgT2,lbl
    def decode_segmap(self,label_mask):
        r,g,b=label_mask.copy(),label_mask.copy(),label_mask.copy()
        for ll in range(0,self.n_classes):
            r[label_mask==ll]=self.color[ll,0]
            g[label_mask==ll]=self.color[ll,1]
            b[label_mask==ll]=self.color[ll,2]
        rgb=np.zeros((label_mask.shape[0],label_mask.shape[1],3))
        rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]=r,g,b
        return rgb
    def lbl_totest(self,pred):
        pred_test=np.zeros((pred.shape[0],pred.shape[1]),np.uint8)
        for ll in range(9):
            pred_test[pred==ll]=self.label_test[ll]
        return pred_test

if __name__=='__main__':
    #path='/media/canpi/DATA/MR_brain/DATA/MRBrainS18/'
    path='../DATA/MRBrainS13DataNii/'
    train_loader=MR13RGBloader_CV(root=path,val_num=1,is_val=False,is_transform=True,is_rotate=True,is_crop=True,is_histeq=True,forest=3)
    val_loader=MR13RGBloader_CV(root=path,val_num=1,is_val=True,is_transform=True,is_rotate=False,is_crop=True,is_histeq=True,forest=3)
    t_loader=data.DataLoader(train_loader, batch_size=1, num_workers=1, shuffle=True)
    v_loader=data.DataLoader(val_loader, batch_size=1, num_workers=1, shuffle=False)
    for i,(region,imgT1,imgIR,imgT2,lbl,edge) in enumerate(t_loader):
        print(i)
        print(region)
        print(imgT1.shape)
        print(imgIR.shape)
        print(imgT2.shape)
        print(lbl.shape)
        print(edge.shape)
        '''
        if i==10:
            cv.imwrite('T1.png',imgT1[1])
            cv.imwrite('IR.png',imgIR[1])
            cv.imwrite('T2.png',imgT2[1])
            cv.imwrite('lbl.png',lbl[1])
            cv.imwrite('edge.png',edge[1])
            print('img saved')
        '''
        print('[{},{},{},{},{}]'.format(imgT1[0,1,40,40],imgIR[0,1,40,40],imgT2[0,1,40,40],lbl[0,1,40,40],edge[0,1,40,40]))

    for i,(region,imgT1,imgIR,imgT2,lbl,edge) in enumerate(v_loader):
        print(i)
        print(region)
        print(imgT1.shape)
        print(imgIR.shape)
        print(imgT2.shape)
        print(lbl.shape)
        print(edge.shape)
        '''
        if i==10:
            cv.imwrite('T1.png',imgT1[1])
            cv.imwrite('IR.png',imgIR[1])
            cv.imwrite('T2.png',imgT2[1])
            cv.imwrite('lbl.png',lbl[1])
            cv.imwrite('edge.png',edge[1])
            print('img saved')
        '''
        print('[{},{},{},{},{}]'.format(imgT1[0,1,40,40],imgIR[0,1,40,40],imgT2[0,1,40,40],lbl[0,1,40,40],edge[0,1,40,40]))
