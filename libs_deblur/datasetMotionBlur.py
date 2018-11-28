import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import ndimage, signal
from scipy import misc
import skimage.transform 
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
import pyblur

        
class Dataset4MotionBlur(Dataset):
    def __init__(self, root_dir, size=[64, 64], set_name='val', sigmaMin=0.5, sigmaMax=2.5, 
                 transform=None, downsampleFactor=1):
        self.root_dir = root_dir
        self.transform = transform
        self.downsampleFactor = downsampleFactor
        if set_name=='val' or set_name=='train': self.set_name = set_name
        else: self.set_name = 'val/'+set_name    
            
        self.current_set_dir = path.join(self.root_dir, self.set_name)             
        self.samplePath = []
        self.sigmaMin, self.sigmaMax = sigmaMin, sigmaMax
        self.size = size        
        self.kernelTransform = transforms.Compose(
            [transforms.ToTensor()             
            ]) 
        
        if set_name=='val':
            for subfolder in os.listdir(self.current_set_dir):
                for sampleFile in os.listdir(path.join(self.current_set_dir, subfolder)):
                    if sampleFile.endswith(("GT.png", "_files")):
                        self.samplePath += [path.join(self.current_set_dir, subfolder, sampleFile)]
        elif set_name == 'train':            
            for subfolder in os.listdir(self.current_set_dir):
                for sampleFile in os.listdir(path.join(self.current_set_dir, subfolder)):
                    if (sampleFile.endswith((".bmp", "_files")) 
                        or sampleFile.endswith((".png","_files")) 
                        or sampleFile.endswith((".jpg", "_files")) ):
                        self.samplePath += [path.join(self.current_set_dir, subfolder, sampleFile)]                    
        else:
            for sampleFile in os.listdir(path.join(self.current_set_dir)):
                if sampleFile.endswith(("GT.png", "_files")):
                    self.samplePath += [path.join(self.current_set_dir, sampleFile)]

        self.current_set_len = len(self.samplePath)   
            
    def __len__(self):        
        return self.current_set_len
    
    
    def __getitem__(self, idx):
        kernelSIZE = 25
        imgNoisyFullRes, imgFullRes = 0,0
        
        imgName = self.samplePath[idx]
        img = misc.imread(imgName)
        img = img.astype(np.float32) # NOT float!!!
        if len(img.shape)<3:
            img = np.expand_dims(img,2)
            img = np.concatenate((img,img,img),2)
        
        
        if self.downsampleFactor<1 and self.set_name!='train':
            img = misc.imresize(img, self.downsampleFactor, 'bicubic')
            img = np.clip(img, 0, 255) 
            img = img.astype(np.float32) 
            #img = np.round(img)
                
        HWC = img.shape         
        th, tw = self.size
        th, tw = th+kernelSIZE*2-1, tw+kernelSIZE*2-1
        if th>0 and tw>0 and self.set_name=='train' and HWC[0]>th and HWC[1]>th: 
            x1 = random.randint(0, HWC[1]-tw)
            y1 = random.randint(0, HWC[0]-th)
            img = img[y1:y1+th,x1:x1+tw,:]
        elif th>0 and tw>0 and HWC[0]>th and HWC[1]>th:
            xcenter = int(HWC[1]/2)
            ycenter = int(HWC[0]/2)
            img = img[ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2),:]        
        
        
        if self.set_name=='train': 
            imgNoisy = np.zeros_like(img)            
            imgNoisy = imgNoisy.astype(np.float32)
            
            #if np.random.rand(1)>0.75:
            #    rescaleFactor = np.random.rand(1)*(1-0.5)+0.5
            #    img = misc.imresize(img, rescaleFactor, 'bicubic')
            #    img = np.clip(img, 0, 255) 
            #    img = img.astype(np.float32)            

            if np.random.rand(1)>0.03:#0.03:
                angle = np.random.randint(0, 180) # counter-clockwise 0~180
                kernelLength = np.random.randint(1, kernelSIZE) # previously set to 20, but can be 1~25, like here as new setup
                xyCenter = int(kernelSIZE/2)
                kernel = np.zeros((kernelSIZE, kernelSIZE))
                kernel[int((kernelSIZE-1)/2), 
                       xyCenter-int((kernelLength-1)/2):xyCenter+int((kernelLength-1)/2)+1] = 1
                kernel = ndimage.rotate(kernel, angle, reshape=False)
                kernel = kernel/np.sum(kernel)

                for i in range(3): 
                    imgNoisy[:,:,i] = signal.convolve2d(img[:,:,i], kernel, mode='same')                            
            else:
                sigma = np.random.rand(1)*(self.sigmaMax-self.sigmaMin)+self.sigmaMin
                kernel = np.zeros((kernelSIZE, kernelSIZE))
                kernel[kernelSIZE//2, kernelSIZE//2] = 1
                kernel = ndimage.gaussian_filter(kernel, sigma[0])
                kernel = kernel/np.sum(kernel)
                for i in range(3): 
                    imgNoisy[:,:,i] = ndimage.gaussian_filter(img[:,:,i], sigma=sigma[0])
                
            imgNoisy = np.clip(imgNoisy, 0, 255) 
            imgNoisy = imgNoisy.astype(np.float32) 
        else:
            kernel = np.zeros((kernelSIZE, kernelSIZE))
            imgNoisy = misc.imread(imgName.replace('GT','blurry'))
            imgNoisy = imgNoisy.astype(np.float32) # NOT float!!!
            #print(imgName.replace('GT','blurry'))
            if len(img.shape)<3:
                imgNoisy = np.expand_dims(imgNoisy,2)
                imgNoisy = np.concatenate((imgNoisy,imgNoisy,imgNoisy),2) 
        
            if self.downsampleFactor<1:
                imgNoisy = misc.imresize(imgNoisy, self.downsampleFactor, 'bicubic')
            #imgNoisy = np.round(imgNoisy)
            imgNoisy = np.clip(imgNoisy, 0, 255)             
            imgNoisy = imgNoisy.astype(np.float32)  
            if th>0 and tw>0:
                imgNoisy = imgNoisy[ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2),:]    
        
        img = img.astype(np.float32)
        imgNoisy = imgNoisy.astype(np.float32)           
        kernel = np.expand_dims(kernel, 2)        
        #HWC = img.shape         
        #th, tw = self.size
        #if th>0 and tw>0 and self.set_name=='train': 
        #    x1 = random.randint(0, HWC[1] - tw)
        #    y1 = random.randint(0, HWC[0] - th)
        #    imgNoisy = imgNoisy[y1:y1+th,x1:x1+tw,:]
        #    img = img[y1:y1+th,x1:x1+tw,:]
        #elif th>0 and tw>0:
        #    xcenter = int(HWC[1]/2)
        #    ycenter = int(HWC[0]/2)                                   
        #    imgNoisy = imgNoisy[ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2),:]
        #    img = img[ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2),:]
        
        if th<0 or tw<0:
            if self.transform:
                img = self.transform(img)
                imgNoisy = self.transform(imgNoisy)        
                kernel = self.kernelTransform(kernel)
            return kernel, imgNoisy, img  
        #print(th,tw)
        HWC = img.shape         
        th, tw = self.size
        xcenter, ycenter = int(HWC[1]/2), int(HWC[0]/2)
        img = img[ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2),:]    
        imgNoisy = imgNoisy[ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2),:]
        #img = img[kernelSIZE:-kernelSIZE+1,kernelSIZE:-kernelSIZE+1,:]
        #imgNoisy = imgNoisy[kernelSIZE:-kernelSIZE+1,kernelSIZE:-kernelSIZE+1,:]
        #print(img.shape,imgNoisy.shape)
        if self.transform:
            img = self.transform(img)
            imgNoisy = self.transform(imgNoisy)        
            kernel = self.kernelTransform(kernel)
                
        return kernel, imgNoisy, img  
        
            
        
    def modcrop(self, image, scale=3):
        """
        To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
        We need to find modulo of height (and width) and scale factor.
        Then, subtract the modulo from height (and width) of original image size.
        There would be no remainder even after scaling operation.
        """
        if len(image.shape) == 3:
            h, w, _ = image.shape
            h = h - np.mod(h, scale)
            w = w - np.mod(w, scale)
            h, w = int(h),int(w)
            image = image[0:h, 0:w, :]
        else:
            h, w = image.shape
            h = h - np.mod(h, scale)
            w = w - np.mod(w, scale)
            h, w = int(h),int(w)
            image = image[0:h, 0:w]
        return image
    
    
    
        
class Dataset4MotionBlur__Evaluation(Dataset):
    def __init__(self, root_dir, size=[64, 64], set_name='val', sigmaMin=0.5, sigmaMax=2.5, 
                 transform=None, downsampleFactor=1):
        self.root_dir = root_dir
        self.transform = transform
        self.downsampleFactor = downsampleFactor
        if set_name=='val' or set_name=='train': self.set_name = set_name
        else: self.set_name = 'val/'+set_name    
            
        self.current_set_dir = path.join(self.root_dir, self.set_name)             
        self.samplePath = []
        self.sigmaMin, self.sigmaMax = sigmaMin, sigmaMax
        self.size = size        
        self.kernelTransform = transforms.Compose(
            [transforms.ToTensor()             
            ]) 
        
        if set_name=='val':
            for subfolder in os.listdir(self.current_set_dir):
                for sampleFile in os.listdir(path.join(self.current_set_dir, subfolder)):
                    if sampleFile.endswith(("GT.png", "_files")):
                        self.samplePath += [path.join(self.current_set_dir, subfolder, sampleFile)]
        elif set_name == 'train':            
            for subfolder in os.listdir(self.current_set_dir):
                for sampleFile in os.listdir(path.join(self.current_set_dir, subfolder)):
                    if (sampleFile.endswith((".bmp", "_files")) 
                        or sampleFile.endswith((".png","_files")) 
                        or sampleFile.endswith((".jpg", "_files")) ):
                        self.samplePath += [path.join(self.current_set_dir, subfolder, sampleFile)]                    
        else:
            for sampleFile in os.listdir(path.join(self.current_set_dir)):
                if sampleFile.endswith(("GT.png", "_files")):
                    self.samplePath += [path.join(self.current_set_dir, sampleFile)]

        self.current_set_len = len(self.samplePath)   
            
    def __len__(self):        
        return self.current_set_len
    
    
    def __getitem__(self, idx):
        kernelSIZE = 25
        imgName = self.samplePath[idx]
        img = misc.imread(imgName)
        img = img.astype(np.float32) # NOT float!!!
        if len(img.shape)<3:
            img = np.expand_dims(img,2)
            img = np.concatenate((img,img,img),2)
        
        imgFullRes = copy.deepcopy(img)
        imgFullRes = imgFullRes.astype(np.float32) 
        if self.downsampleFactor<1 and self.set_name!='train':
            imgFullRes = copy.deepcopy(img)
            img = misc.imresize(img, self.downsampleFactor, 'bicubic')
            img = np.clip(img, 0, 255) 
            img = img.astype(np.float32) 
            #img = np.round(img)
                
        HWC = img.shape         
        th, tw = self.size
        th, tw = th+kernelSIZE*2-1, tw+kernelSIZE*2-1
        if th>0 and tw>0 and self.set_name=='train' and HWC[0]>th and HWC[1]>th: 
            x1 = random.randint(0, HWC[1]-tw)
            y1 = random.randint(0, HWC[0]-th)
            img = img[y1:y1+th,x1:x1+tw,:]
        elif th>0 and tw>0 and HWC[0]>th and HWC[1]>th:
            xcenter = int(HWC[1]/2)
            ycenter = int(HWC[0]/2)
            img = img[ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2),:]        
        
        
        if self.set_name=='train': 
            imgNoisy = np.zeros_like(img)            
            imgNoisy = imgNoisy.astype(np.float32)
            
            #if np.random.rand(1)>0.75:
            #    rescaleFactor = np.random.rand(1)*(1-0.5)+0.5
            #    img = misc.imresize(img, rescaleFactor, 'bicubic')
            #    img = np.clip(img, 0, 255) 
            #    img = img.astype(np.float32)            

            if np.random.rand(1)>0.03:#0.03:
                angle = np.random.randint(0, 180) # counter-clockwise 0~180
                kernelLength = np.random.randint(1, 20) # 1~25
                xyCenter = int(kernelSIZE/2)
                kernel = np.zeros((kernelSIZE, kernelSIZE))
                kernel[int((kernelSIZE-1)/2), 
                       xyCenter-int((kernelLength-1)/2):xyCenter+int((kernelLength-1)/2)+1] = 1
                kernel = ndimage.rotate(kernel, angle, reshape=False)
                kernel = kernel/np.sum(kernel)

                for i in range(3): 
                    imgNoisy[:,:,i] = signal.convolve2d(img[:,:,i], kernel, mode='same')                            
            else:
                sigma = np.random.rand(1)*(self.sigmaMax-self.sigmaMin)+self.sigmaMin
                kernel = np.zeros((kernelSIZE, kernelSIZE))
                kernel[kernelSIZE//2, kernelSIZE//2] = 1
                kernel = ndimage.gaussian_filter(kernel, sigma[0])
                kernel = kernel/np.sum(kernel)
                for i in range(3): 
                    imgNoisy[:,:,i] = ndimage.gaussian_filter(img[:,:,i], sigma=sigma[0])
                
            imgNoisy = np.clip(imgNoisy, 0, 255) 
            imgNoisy = imgNoisy.astype(np.float32) 
        else:
            kernel = np.zeros((kernelSIZE, kernelSIZE))
            imgNoisy = misc.imread(imgName.replace('GT','blurry'))
            imgNoisy = imgNoisy.astype(np.float32) # NOT float!!!
            #print(imgName.replace('GT','blurry'))
            if len(img.shape)<3:
                imgNoisy = np.expand_dims(imgNoisy,2)
                imgNoisy = np.concatenate((imgNoisy,imgNoisy,imgNoisy),2) 
        
            imgNoisyFullRes = copy.deepcopy(imgNoisy)
            imgNoisyFullRes = imgNoisyFullRes.astype(np.float32)  
            if self.downsampleFactor<1:
                imgNoisy = misc.imresize(imgNoisy, self.downsampleFactor, 'bicubic')
            #imgNoisy = np.round(imgNoisy)
            imgNoisy = np.clip(imgNoisy, 0, 255)             
            imgNoisy = imgNoisy.astype(np.float32)  
            if th>0 and tw>0:
                imgNoisy = imgNoisy[ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2),:]    
        
        img = img.astype(np.float32)
        imgNoisy = imgNoisy.astype(np.float32)           
        kernel = np.expand_dims(kernel, 2)        
        #HWC = img.shape         
        #th, tw = self.size
        #if th>0 and tw>0 and self.set_name=='train': 
        #    x1 = random.randint(0, HWC[1] - tw)
        #    y1 = random.randint(0, HWC[0] - th)
        #    imgNoisy = imgNoisy[y1:y1+th,x1:x1+tw,:]
        #    img = img[y1:y1+th,x1:x1+tw,:]
        #elif th>0 and tw>0:
        #    xcenter = int(HWC[1]/2)
        #    ycenter = int(HWC[0]/2)                                   
        #    imgNoisy = imgNoisy[ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2),:]
        #    img = img[ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2),:]
        
        if th<0 or tw<0:
            if self.transform:
                img = self.transform(img)
                imgNoisy = self.transform(imgNoisy)        
                kernel = self.kernelTransform(kernel)
            return kernel, imgNoisyFullRes, imgFullRes, imgNoisy, img  
        #print(th,tw)
        HWC = img.shape         
        th, tw = self.size
        xcenter, ycenter = int(HWC[1]/2), int(HWC[0]/2)
        img = img[ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2),:]    
        imgNoisy = imgNoisy[ycenter-int(th/2):ycenter+int(th/2),xcenter-int(tw/2):xcenter+int(tw/2),:]
        #img = img[kernelSIZE:-kernelSIZE+1,kernelSIZE:-kernelSIZE+1,:]
        #imgNoisy = imgNoisy[kernelSIZE:-kernelSIZE+1,kernelSIZE:-kernelSIZE+1,:]
        #print(img.shape,imgNoisy.shape)
        if self.transform:
            img = self.transform(img)
            imgNoisy = self.transform(imgNoisy)        
            kernel = self.kernelTransform(kernel)
                
        return kernel, imgNoisyFullRes, imgFullRes, imgNoisy, img  
            