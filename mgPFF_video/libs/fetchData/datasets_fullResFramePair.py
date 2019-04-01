import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

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
    def __init__(self, root_dir, size=[256, 256], set_name='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.set_name = set_name
        self.current_set_dir = path.join(self.root_dir, self.set_name)        
        self.current_set_len = len(os.listdir(self.current_set_dir))-2
        self.size = size
        
    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):             
        if idx==0: idx+=1
        
        idx = min(self.current_set_len-1,idx)
        if self.set_name=='train': 
            idx += 120              
            imgName1 = 'SintelFrame'+format(idx,'05d')+'.png' # str(curBatchIndex)  000003
            imgName1 = path.join(self.current_set_dir, imgName1)
            imgName2 = 'SintelFrame'+format(idx+1,'05d')+'.png' # str(curBatchIndex)  000004
            imgName2 = path.join(self.current_set_dir, imgName2)
        else:
            imgName1 = 'frame_'+format(idx,'04d')+'.png' # str(curBatchIndex)  000003
            imgName1 = path.join(self.current_set_dir, imgName1)
            imgName2 = 'frame_'+format(idx+1,'04d')+'.png' # str(curBatchIndex)  000004
            imgName2 = path.join(self.current_set_dir, imgName2)


        #image1 = Image.open(imgName1)
        #image2 = Image.open(imgName2)
        image1 = misc.imread(imgName1)
        image2 = misc.imread(imgName2)
        image1 = image1.astype(np.float32)            
        image2 = image2.astype(np.float32)            
        
        
        if self.set_name=='train' and np.random.random(1)>0.5:
            image1 = np.flip(image1,axis=0).copy()
            image2 = np.flip(image2,axis=0).copy()            
        if self.set_name=='train' and np.random.random(1)>0.5:
            image1 = np.flip(image1,axis=1).copy()
            image2 = np.flip(image2,axis=1).copy()
        if self.set_name=='train':
            k = np.random.randint(4)
            image1 = np.rot90(image1, k).copy()
            image2 = np.rot90(image2, k).copy()      
        if len(image1.shape)==2:
            image1 = np.expand_dims(image1, axis=2)
            image1 = np.repeat(image1, 3, axis=2)
        if len(image2.shape)==2:
            image2 = np.expand_dims(image2, axis=2)
            image2 = np.repeat(image2, 3, axis=2)
        
        
        if self.transform: 
            try:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
            except:
                print(imgName1,imgName2, image1.shape, image2.shape)

        sample = (imgName1, imgName2, image1, image2)
        if self.set_name=='train' and self.size[0]>0 and self.size[1]>0: 
            CHW = image1.size() 
            th, tw = self.size
            th = min(th, image1.size(1))
            tw = min(tw, image1.size(2))
            x1 = random.randint(0, CHW[2] - tw)
            y1 = random.randint(0, CHW[1] - th)
            sample = (imgName1, imgName2, 
                      image1[:,y1:y1+th,x1:x1+tw], image2[:,y1:y1+th,x1:x1+tw])
        elif self.set_name=='val' and self.size[0]>0 and self.size[1]>0: 
            CHW = image1.size() 
            th, tw = self.size
            th = min(th, image1.size(1))
            tw = min(tw, image1.size(2))
            xcenter = int(CHW[2]/2)
            ycenter = int(CHW[1]/2)            
            sample = (imgName1, imgName2,
                      image1[:,ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2)], 
                      image2[:,ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2)])
        return sample
        
    
    def imshow(self, img):
        #img = img / 2 + 0.5    
        img -= img.min()
        img /= img.max()
        npimg = img.detach().squeeze(-1)
        #npimg = img.numpy()
        npimg = np.transpose(npimg,(1,2,0))
        npimg = np.clip(npimg,0,1)
        plt.imshow(npimg)