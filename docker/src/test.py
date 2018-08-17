import numpy as np
import nibabel as nib
import cv2 as cv
import torch
from torch.utils import data
from torchvision.transforms import transforms
from preprocess import readVol,to_uint8,IR_to_uint8,histeq,preprocessed,get_stacked,rotate,calc_crop_region,calc_max_region_list,crop,get_edge

import os
from torch.autograd import Variable
from fcn_xu import fcn_mul

inputDir='/input'
outputDir='/output'
T1_path=os.path.join(inputDir, 'pre', 'reg_T1.nii.gz')
IR_path=os.path.join(inputDir, 'pre', 'reg_IR.nii.gz')
T2_path=os.path.join(inputDir, 'pre', 'FLAIR.nii.gz')
resultpath=os.path.join(outputDir, 'result.nii.gz')

model_1_path='/models/val_1_e_100.pkl'
model_2_path='/models/val_2_e_100.pkl'
model_3_path='/models/val_3_e_100.pkl'
model_4_path='/models/val_4_e_100.pkl'
model_5_path='/models/val_5_e_400.pkl'
model_6_path='/models/val_6_e_400.pkl'
model_7_path='/models/val_7_e_400.pkl'

class MR18loader_test(data.Dataset):
    def __init__(self,T1_path,IR_path,T2_path,is_transform,is_crop,is_hist,forest):
        self.T1_path=T1_path
        self.IR_path=IR_path
        self.T2_path=T2_path
        self.is_transform=is_transform
        self.is_crop=is_crop
        self.is_hist=is_hist
        self.forest=forest
        self.n_classes=11 
        self.T1mean=0.0
        self.IRmean=0.0
        self.T2mean=0.0
        #read data
        T1_nii=to_uint8(readVol(self.T1_path))
        IR_nii=IR_to_uint8(readVol(self.IR_path))
        T2_nii=to_uint8(readVol(self.T2_path))
        #histeq
        if self.is_hist:
            T1_nii=histeq(T1_nii)
        #stack 
        T1_stack_list=get_stacked(T1_nii,self.forest)
        IR_stack_list=get_stacked(IR_nii,self.forest)
        T2_stack_list=get_stacked(T2_nii,self.forest)
        #crop
        if self.is_crop:
            region_list=calc_max_region_list(calc_crop_region(T1_stack_list,50,5),self.forest)
            self.region_list=region_list
            T1_stack_list=crop(T1_stack_list,region_list)
            IR_stack_list=crop(IR_stack_list,region_list)
            T2_stack_list=crop(T2_stack_list,region_list)
        #get mean
        '''
        T1mean,IRmean,T2mean=0.0,0.0,0.0
        for samples in T1_stack_list:
            for stacks in samples:
                T1mean=T1mean+np.mean(stacks)
        self.T1mean=T1mean/(len(T1_stack_list)*len(T1_stack_list[0]))
        for samples in IR_stack_list:
            for stacks in samples:
                IRmean=IRmean+np.mean(stacks)
        self.IRmean=IRmean/(len(IR_stack_list)*len(IR_stack_list[0]))
        for samples in T2_stack_list:
            for stacks in samples:
                T2mean=T2mean+np.mean(stacks)
        self.T2mean=T2mean/(len(T2_stack_list)*len(T2_stack_list[0]))
        '''
        self.T1mean=94.661544495
        self.IRmean=88.574705283
        self.T2mean=32.376038631

        #transform
        if self.is_transform:
            for stack_index in range(len(T1_stack_list)):
                T1_stack_list[stack_index],  \
                IR_stack_list[stack_index],  \
                T2_stack_list[stack_index]=  \
                self.transform(              \
                T1_stack_list[stack_index],  \
                IR_stack_list[stack_index],  \
                T2_stack_list[stack_index]) 

        # data ready
        self.T1_stack_list=T1_stack_list
        self.IR_stack_list=IR_stack_list
        self.T2_stack_list=T2_stack_list

    def __len__(self):
        return 48
    def __getitem__(self,index):
        return self.region_list[index],self.T1_stack_list[index],self.IR_stack_list[index],self.T2_stack_list[index]
    
    def transform(self,imgT1,imgIR,imgT2):
        imgT1=torch.from_numpy((imgT1.transpose(2,0,1).astype(np.float)-self.T1mean)/255.0).float()
        imgIR=torch.from_numpy((imgIR.transpose(2,0,1).astype(np.float)-self.IRmean)/255.0).float()
        imgT2=torch.from_numpy((imgT2.transpose(2,0,1).astype(np.float)-self.T2mean)/255.0).float()
        return imgT1,imgIR,imgT2

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
    #io vols
    srcvol=nib.load(T1_path)
    outvol=np.zeros((240,240,48),np.uint8)
    #data loader
    loader=MR18loader_test(T1_path=T1_path,IR_path=IR_path,T2_path=T2_path,is_transform=True,is_crop=True,is_hist=True,forest=3)
    testloader=data.DataLoader(loader,batch_size=1,num_workers=1,shuffle=False)
    #model setup
    n_classes = loader.n_classes
    model_1=fcn_mul(n_classes=n_classes)
    model_2=fcn_mul(n_classes=n_classes)
    model_3=fcn_mul(n_classes=n_classes)
    model_4=fcn_mul(n_classes=n_classes)
    model_5=fcn_mul(n_classes=n_classes)
    model_6=fcn_mul(n_classes=n_classes)
    model_7=fcn_mul(n_classes=n_classes)
    model_1.cuda()
    model_2.cuda()
    model_3.cuda()
    model_4.cuda()
    model_5.cuda()
    model_6.cuda()
    model_7.cuda()
    state_1 = torch.load(model_1_path)['model_state']
    state_2 = torch.load(model_2_path)['model_state']
    state_3 = torch.load(model_3_path)['model_state']
    state_4 = torch.load(model_4_path)['model_state']
    state_5 = torch.load(model_5_path)['model_state']
    state_6 = torch.load(model_6_path)['model_state']
    state_7 = torch.load(model_7_path)['model_state']
    model_1.load_state_dict(state_1)
    model_2.load_state_dict(state_2)
    model_3.load_state_dict(state_3)
    model_4.load_state_dict(state_4)
    model_5.load_state_dict(state_5)
    model_6.load_state_dict(state_6)
    model_7.load_state_dict(state_7)
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    model_5.eval()
    model_6.eval()
    model_7.eval()
    #test
    for i_t,(regions_t,T1s_t,IRs_t,T2s_t) in enumerate(testloader):
        T1s_t,IRs_t,T2s_t=Variable(T1s_t.cuda()),Variable(IRs_t.cuda()),Variable(T2s_t.cuda())
        with torch.no_grad():
            out_1=model_1(T1s_t,IRs_t,T2s_t)[0,:,:,:]
            out_2=model_2(T1s_t,IRs_t,T2s_t)[0,:,:,:]
            out_3=model_3(T1s_t,IRs_t,T2s_t)[0,:,:,:]
            out_4=model_4(T1s_t,IRs_t,T2s_t)[0,:,:,:]
            out_5=model_5(T1s_t,IRs_t,T2s_t)[0,:,:,:]
            out_6=model_6(T1s_t,IRs_t,T2s_t)[0,:,:,:]
            out_7=model_7(T1s_t,IRs_t,T2s_t)[0,:,:,:]
        pred_1 = out_1.data.max(0)[1].cpu().numpy()
        pred_2 = out_2.data.max(0)[1].cpu().numpy()
        pred_3 = out_3.data.max(0)[1].cpu().numpy()
        pred_4 = out_4.data.max(0)[1].cpu().numpy()
        pred_5 = out_5.data.max(0)[1].cpu().numpy()
        pred_6 = out_6.data.max(0)[1].cpu().numpy()
        pred_7 = out_7.data.max(0)[1].cpu().numpy()
        h,w=pred_1.shape[0],pred_1.shape[1]
        pred=np.zeros((h,w),np.uint8)
        for y in range(h):
            for x in range(w):
                pred_list=np.array([pred_1[y,x],pred_2[y,x],pred_3[y,x],pred_4[y,x],pred_5[y,x],pred_6[y,x],pred_7[y,x]])
                pred[y,x]=np.argmax(np.bincount(pred_list))
        pred_pad=np.zeros((240,240),np.uint8)
        pred_pad[regions_t[0]:regions_t[1],regions_t[2]:regions_t[3]]=pred[0:regions_t[1]-regions_t[0],0:regions_t[3]-regions_t[2]]
        #remove noise
        mask=np.zeros((240,240),np.uint8)
        mask=(pred_pad>0).astype(np.uint8)
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(5, 5))
        mask = cv.erode(mask,kernel)
        mask = cv.dilate(mask,kernel)
        pred_pad=pred_pad*mask
        outvol[:,:,i_t]=pred_pad.transpose()
    nib.Nifti1Image(outvol, srcvol.affine, srcvol.header).to_filename(resultpath)

