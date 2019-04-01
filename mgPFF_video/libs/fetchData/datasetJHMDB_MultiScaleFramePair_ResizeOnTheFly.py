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

class JHMDBFramePair(Dataset):
    def __init__(self, root_dir, downsizeFactorList=[1,1/2.,1/4.,1/8.,1/16.], 
                 size=[384, 768], set_name='train', TFNormalize=None):
        self.root_dir = root_dir
        self.transform = transform
        self.set_name = set_name
        self.current_set_dir = path.join(self.root_dir, self.set_name)
        self.current_set_len = 0 #len(os.listdir(self.current_set_dir))        
        self.size = size    
        self.imageformat = '.png'
        
        self.samplePath = []
        for action in os.listdir(self.current_set_dir): 
            if action[0]=='.': continue
            for video in os.listdir(path.join(self.current_set_dir, action)):
                if video[0]=='.': continue
                set_path = []
                frameID = 1            
                for frame in os.listdir(path.join(self.current_set_dir, action, video)): # self.sorted_dir
                    if frame.endswith((".png","_files")) or frame.endswith((".jpg","_files")):
                        set_path += [path.join(self.current_set_dir, action, 
                                               video, format(frameID,'05d'))]                        
                        frameID+=1
                set_path.sort()
                if self.set_name=='train': set_path = set_path[:-3] # enables larger displacement during training
                else: set_path = set_path[:-1]
                self.samplePath += set_path
        self.current_set_len = len(self.samplePath)
        
        
        self.downsizeFactorList = downsizeFactorList
        self.TF2tensor = transforms.ToTensor()
        self.TF2PIL = transforms.ToPILImage()        
        self.TFNormalize = TFNormalize # transforms.Normalize((127.,127.,127.),(127.,127.,127.))
        self.TFResizeList = [] # transforms.Resize((24, 64))
        for i in range(len(self.downsizeFactorList)):
            self.TFResizeList += [transforms.Resize( 
                (int(self.size[0]*self.downsizeFactorList[i]), int(self.size[1]*self.downsizeFactorList[i]))) ]
        
        self.tmpTFresize = transforms.Resize((self.size[0],self.size[1]))

    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):
        imgName1 = self.samplePath[idx]+self.imageformat
        curpath, frameID = os.path.split(imgName1)
        frameID = int(frameID.replace(self.imageformat,''))
        if self.set_name=='train': frameStride = random.randint(1,3)
        else: frameStride = 1
        
        imgName2 = path.join(curpath, format(frameID+frameStride,'05d'))+self.imageformat
        
        image1 = PIL.Image.open(imgName1)
        image2 = PIL.Image.open(imgName2)
                
        image1 = self.tmpTFresize(image1)
        image2 = self.tmpTFresize(image2)
        
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
        
        
        
        
        
        
        
class DAVISFramePair4ThatVideo(Dataset):
    def __init__(self, root_dir, downsizeFactorList=[1,1/2.,1/4.,1/8.,1/16.], 
                 size=[384, 768], set_name='train', TFNormalize=None):
        self.root_dir = root_dir
        self.transform = transform
        self.set_name = set_name
        self.current_set_dir = self.root_dir
        self.current_set_len = len(os.listdir(self.current_set_dir))        
        self.size = size               
        
        self.samplePath = []
        set_path = []
        frameID = 0
        for sampleFile in os.listdir(self.current_set_dir): # self.sorted_dir
            if sampleFile.endswith((".jpg","_files")):
                set_path += [path.join(self.current_set_dir, format(frameID,'05d'))]
                frameID+=1
        set_path.sort()
        if self.set_name=='train': set_path = set_path[:-3] # enables larger displacement during training
        else: set_path = set_path[:-1]
        self.samplePath += set_path
        self.current_set_len = len(self.samplePath)
        
        
        self.downsizeFactorList = downsizeFactorList
        self.TF2tensor = transforms.ToTensor()
        self.TF2PIL = transforms.ToPILImage()        
        self.TFNormalize = TFNormalize # transforms.Normalize((127.,127.,127.),(127.,127.,127.))
        self.TFResizeList = [] # transforms.Resize((24, 64))
        for i in range(len(self.downsizeFactorList)):
            self.TFResizeList += [transforms.Resize( 
                (int(self.size[0]*self.downsizeFactorList[i]), int(self.size[1]*self.downsizeFactorList[i]))) ]
        
        self.tmpTFresize = transforms.Resize((self.size[0],self.size[1]))

    def __len__(self):        
        return self.current_set_len
    
    def __getitem__(self, idx):
        imgName1 = self.samplePath[idx]+'.jpg'
        curpath, frameID = os.path.split(imgName1)
        frameID = int(frameID.replace('.jpg',''))
        if self.set_name=='train': frameStride = random.randint(1,3)
        else: frameStride = 1
        
        imgName2 = path.join(curpath, format(frameID+frameStride,'05d'))+'.jpg'
        
        image1 = PIL.Image.open(imgName1)
        image2 = PIL.Image.open(imgName2)
                
        image1 = self.tmpTFresize(image1)
        image2 = self.tmpTFresize(image2)
        
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