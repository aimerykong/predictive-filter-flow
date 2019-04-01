import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import misc
import matplotlib.pyplot as plt
import PIL.Image

import skimage.transform 


import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms

class MPISintelFramePair(Dataset):
    def __init__(self, root_dir, downsizeFactorList=[1,1/2.,1/4.,1/8.,1/16.], 
                 size=[256, 256], set_name='train', TFNormalize=None):
        self.root_dir = root_dir
        self.transform = transform
        self.set_name = set_name
        self.current_set_dir = path.join(self.root_dir, self.set_name)      
        self.current_set_len = len(os.listdir(self.current_set_dir))-2
        if self.set_name=='train': self.current_set_len -= 3
        self.size = size
        
        self.downsizeFactorList = downsizeFactorList
        self.TF2tensor = transforms.ToTensor()
        self.TF2PIL = transforms.ToPILImage()        
        self.TFNormalize = TFNormalize # transforms.Normalize((127.,127.,127.),(127.,127.,127.))
        self.TFResizeList = [] # transforms.Resize((24, 64))
        for i in range(len(self.downsizeFactorList)):
            self.TFResizeList += [transforms.Resize( 
                (int(self.size[0]*self.downsizeFactorList[i]), int(self.size[1]*self.downsizeFactorList[i]))) ]
        
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):             
        #if idx==0: idx+=1
        idx+=1
        idx = min(self.current_set_len-1,idx)
        if self.set_name=='train': idx = min(self.current_set_len-1-3,idx)
            
        if self.set_name=='train': 
            idx += 120              
            imgName1 = 'SintelFrame'+format(idx,'05d')+'.png' # str(curBatchIndex)  000003
            imgName1 = path.join(self.current_set_dir, imgName1)
            stride=random.randint(1,3)
            imgName2 = 'SintelFrame'+format(idx+stride,'05d')+'.png' # str(curBatchIndex)  000004
            imgName2 = path.join(self.current_set_dir, imgName2)
        else:
            imgName1 = 'frame_'+format(idx,'04d')+'.png' # str(curBatchIndex)  000003
            imgName1 = path.join(self.current_set_dir, imgName1)                        
            imgName2 = 'frame_'+format(idx+1,'04d')+'.png' # str(curBatchIndex)  000004
            imgName2 = path.join(self.current_set_dir, imgName2)


        image1 = PIL.Image.open(imgName1)
        image2 = PIL.Image.open(imgName2)
        #image1 = image1.astype(np.float32)            
        #image2 = image2.astype(np.float32)        
        
        if self.set_name=='train' and np.random.random(1)>0.5:
            image1 = image1.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            image2 = image2.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        if self.set_name=='train' and np.random.random(1)>0.5:
            image1 = image1.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            image2 = image2.transpose(PIL.Image.FLIP_LEFT_RIGHT)

            
        image1 = self.TF2tensor(image1)
        image2 = self.TF2tensor(image2)
        if image1.size(0)==1:
            image1 = image1.expand(3,image1.size(1),image1.size(2))
        if image2.size(0)==1:
            image2 = image2.expand(3,image2.size(1),image2.size(2))
        
        CHW = image1.size()
        if self.size[0]<=0 or self.size[1]<=0:
            self.size = [CHW[1],CHW[2]]
            self.TFResizeList = [] # transforms.Resize((24, 64))
            for i in range(len(self.downsizeFactorList)):
                self.TFResizeList += [transforms.Resize( 
                    (int(self.size[0]*self.downsizeFactorList[i]), int(self.size[1]*self.downsizeFactorList[i]))) ]
        
        
        
        elif self.set_name=='train' and self.size[0]>0 and self.size[1]>0: 
            CHW = image1.size()             
            th, tw = self.size
            th = min(th, image1.size(1))
            tw = min(tw, image1.size(2))            
            #print(self.size, CHW, th, tw)            
            x1 = random.randint(0, CHW[2] - tw)
            y1 = random.randint(0, CHW[1] - th)
            image1 = image1[:,y1:y1+th,x1:x1+tw]
            image2 = image2[:,y1:y1+th,x1:x1+tw]
        elif self.set_name=='val' and self.size[0]>0 and self.size[1]>0: 
            CHW = image1.size() 
            th, tw = self.size
            th = min(th, image1.size(1))
            tw = min(tw, image1.size(2))
            xcenter = int(CHW[2]/2)
            ycenter = int(CHW[1]/2)            
            image1 = image1[:,ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2)] 
            image2 = image2[:,ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2)]
            
        H, W = CHW[1:]
        image1 = self.TF2PIL(image1)
        image2 = self.TF2PIL(image2)
        sampleList = []
        for curScaleIdx in range(len(self.downsizeFactorList)):
            if self.downsizeFactorList[curScaleIdx]==1:
                curImage = self.TF2tensor(image1)
                curImage = self.TFNormalize(curImage)
                #print(self.downsizeFactorList[curScaleIdx], curImage.shape, curScaleIdx)
                sampleList += [curImage]
                curImage = self.TF2tensor(image2)
                curImage = self.TFNormalize(curImage)
                #print(curImage.shape, curScaleIdx)
                sampleList += [curImage]
            else:
                curImage = self.TFResizeList[curScaleIdx](image1)
                curImage = self.TF2tensor(curImage)
                curImage = self.TFNormalize(curImage)
                #print(curImage.shape, curScaleIdx)
                #print(self.downsizeFactorList[curScaleIdx], curImage.shape, curScaleIdx)
                sampleList += [curImage]

                curImage = self.TFResizeList[curScaleIdx](image2)
                curImage = self.TF2tensor(curImage)
                curImage = self.TFNormalize(curImage)
                #print(curImage.shape, curScaleIdx)
                sampleList += [curImage]
        #print('\n')        
        return tuple(sampleList)
    
        
    
    def imshow(self, img):
        #img = img / 2 + 0.5    
        img -= img.min()
        img /= img.max()
        npimg = img.detach().squeeze(-1)
        #npimg = img.numpy()
        npimg = np.transpose(npimg,(1,2,0))
        npimg = np.clip(npimg,0,1)
        plt.imshow(npimg)