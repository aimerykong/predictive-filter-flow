import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms


    
class Loss4GridSamplingWithFlowVecByFilterFlow_OcclusionAware(nn.Module):
    def __init__(self, device='cpu', weight=1):
        super(Loss4GridSamplingWithFlowVecByFilterFlow_OcclusionAware, self).__init__()
        self.device = device
        self.reconstructImage = 0
        self.diff = 0
        self.epsilon = 0.001
        self.weight = weight
        
    def forward(self, image1, image2, filterFlow_img1_to_img2, validMask=torch.tensor([])):
        self.UVgrid = self.filterFlow2UV(filterFlow_img1_to_img2)
        
        N,C,H,W = image1.size()        
        self.reconstructImage, mask = self.funcOpticalFlowWarp(image1, self.UVgrid, self.device)
        if max(validMask.shape):
            mask = validMask*mask
        
        self.diff = self.reconstructImage - image2              
        self.diff = torch.sqrt(self.diff**2+self.epsilon**2)
        self.diff = mask*self.diff
        totloss = torch.sum(torch.sum(torch.sum(torch.sum(self.diff))))        
        totloss = totloss/(N*C*H*W)        
        return totloss*self.weight
    
    
    def filterFlow2UV(self, offsetTensor): # in pytorch tensor format
        kernelSize = offsetTensor.size(1)**0.5
        
        if kernelSize%2==1:    
            kernelSize = int(offsetTensor.size(1)**0.5/2)            
            yv, xv = torch.meshgrid([torch.arange(-kernelSize,kernelSize+1),
                                     torch.arange(-kernelSize,kernelSize+1)])
        else:    
            kernelSize = int(offsetTensor.size(1)**0.5/2)            
            yv, xv = torch.meshgrid([torch.arange(-kernelSize,kernelSize),
                                     torch.arange(-kernelSize,kernelSize)])

        yv, xv = yv.unsqueeze(0).type('torch.FloatTensor'), xv.unsqueeze(0).type('torch.FloatTensor')

        yv = yv.contiguous().view(1,-1)
        yv = yv.unsqueeze(-1).unsqueeze(-1).to(self.device)
        yv = Variable(yv)
        flowMapY = offsetTensor*yv
        flowMapY = torch.sum(flowMapY,1)
        flowMapY = flowMapY.unsqueeze(1)

        xv = xv.contiguous().view(1,-1)
        xv = xv.unsqueeze(-1).unsqueeze(-1).to(self.device)    
        xv = Variable(xv)
        flowMapX = offsetTensor*xv # x
        flowMapX = torch.sum(flowMapX,1)
        flowMapX = flowMapX.unsqueeze(1)

        return torch.cat([flowMapX,flowMapY],1) # [x,y]
    
    
    def funcOpticalFlowWarp(self, x, flo, device='cpu'):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        grid = grid.to(device)
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
        mask = F.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        return output*mask, mask
    
    
    
class Loss4RobustPixelReconstruction_OcclusionAware(nn.Module):
    def __init__(self, device='cpu', filterSize=11, dilate=1):
        super(Loss4RobustPixelReconstruction_OcclusionAware, self).__init__()
        self.device = device
        self.filterSize = filterSize        
        self.filterSize2Channel = self.filterSize**2
        self.reconstructImage = 0
        self.dilate = dilate
        self.newFilterSize = (self.filterSize-1)*(self.dilate-1)+self.filterSize
        if self.newFilterSize%2==1:
            self.padSize = int(self.newFilterSize/2)
        else: 
            self.padSize = [int(self.newFilterSize/2)-1,  # left, right, top, bottom
                            int(self.newFilterSize/2), 
                            int(self.newFilterSize/2)-1, 
                            int(self.newFilterSize/2)]
        self.diff = 0
        self.epsilon = 0.001
        
    def forward(self, image1, image2, filters_img1_to_img2, validMask=torch.tensor([]), mode='train'):
        N,C,H,W = image1.size()
        self.reconstructImage = self.rgbImageFilterFlow(image1, filters_img1_to_img2)
        self.diff = self.reconstructImage - image2               
        #self.diff = torch.abs(self.diff)
        self.diff = torch.sqrt(self.diff**2+self.epsilon**2)        
        if max(validMask.shape):
            self.diff = self.diff*validMask
            
        totloss = torch.sum(torch.sum(torch.sum(torch.sum(self.diff))))        
        totloss = totloss/(N*C*H*W)
        return totloss
    
    def rgbImageFilterFlow(self, img, filters):                
        inputChannelSize = 1
        outputChannelSize = 1
        N = img.size(0)
    
        #paddingFunc = nn.ZeroPad2d(int(self.filterSize/2))
        paddingFunc = nn.ZeroPad2d(self.padSize)
        img = paddingFunc(img)        
        imgSize = [img.size(2),img.size(3)]
        
        out_R = F.unfold(img[:,0,:,:].unsqueeze(1), (self.filterSize, self.filterSize), self.dilate)
        out_R = out_R.view(N, out_R.size(1), 
                           imgSize[0]-self.filterSize+1, imgSize[1]-self.filterSize+1)    
        #out_R = paddingFunc(out_R)
        out_R = torch.mul(out_R, filters)
        out_R = torch.sum(out_R, dim=1).unsqueeze(1)

        out_G = F.unfold(img[:,1,:,:].unsqueeze(1), (self.filterSize, self.filterSize), self.dilate)
        out_G = out_G.view(N, out_G.size(1), 
                           imgSize[0]-self.filterSize+1, imgSize[1]-self.filterSize+1)    
        #out_G = paddingFunc(out_G)
        out_G = torch.mul(out_G, filters)
        out_G = torch.sum(out_G, dim=1).unsqueeze(1)

        out_B = F.unfold(img[:,2,:,:].unsqueeze(1), (self.filterSize, self.filterSize), self.dilate)
        out_B = out_B.view(N, out_B.size(1), 
                           imgSize[0]-self.filterSize+1, imgSize[1]-self.filterSize+1)    
        #out_B = paddingFunc(out_B)
        out_B = torch.mul(out_B, filters)
        out_B = torch.sum(out_B, dim=1).unsqueeze(1)
        return torch.cat([out_R, out_G, out_B], 1)
    
    

class Loss4ImageGradientDifference_OcclusionAware(nn.Module):
    def __init__(self, device='cpu', weight=1):
        super(Loss4ImageGradientDifference_OcclusionAware, self).__init__()
        self.device = device
        self.weight = weight
        self.hMapImage1 = 0
        self.hMapImage2 = 0
        self.vMapImage1 = 0
        self.vMapImage2 = 0
        self.epsilon = 0.001
        
    def forward(self, image1, image2, validMask=torch.tensor([])):
        N,C,H,W = image1.size()
        
        horizontalGradient = nn.Conv2d(in_channels=C, out_channels=C, 
                                       kernel_size=(1,3), stride=1, padding=0, bias=False,
                                       groups=C)
        verticalGradient = nn.Conv2d(in_channels=C, out_channels=C,
                                     kernel_size=(3,1), stride=1, padding=0, bias=False,
                                     groups=C)
        
        hKernel = torch.Tensor([[-1, 2, -1]]).unsqueeze(0).unsqueeze(0)
        hKernel = hKernel.expand(horizontalGradient.weight.size()).to(self.device)
        horizontalGradient.weight = torch.nn.Parameter(hKernel, requires_grad=False)
        
        vKernel = torch.Tensor([[-1], [2], [-1]]).unsqueeze(0).unsqueeze(0)
        vKernel = vKernel.expand(verticalGradient.weight.size()).to(self.device)
        verticalGradient.weight = torch.nn.Parameter(vKernel, requires_grad=False)
        
        self.hMapImage1 = horizontalGradient(image1)
        self.hMapImage2 = horizontalGradient(image2)        
        hh, ww = self.hMapImage1.size(2), self.hMapImage1.size(3)
        hloss = torch.sqrt((self.hMapImage1-self.hMapImage2)**2+self.epsilon**2)
        if max(validMask.shape):
            hloss = hloss*validMask
        hloss = torch.sum(torch.sum(torch.sum(torch.sum(hloss))))/(N*C*H*W)
        
        
        self.vMapImage1 = verticalGradient(image1)
        self.vMapImage2 = verticalGradient(image2)
        hh, ww = self.vMapImage1.size(2), self.vMapImage1.size(3)
        vloss = torch.sqrt((self.vMapImage1-self.vMapImage2)**2+self.epsilon**2)
        if max(validMask.shape):
            vloss = vloss*validMask
        vloss = torch.sum(torch.sum(torch.sum(torch.sum(vloss))))/(N*C*H*W)
        
        
        return (vloss+hloss)*self.weight   
    

    
class Loss4BidirFlowVec_OcclusionAware(nn.Module):
    def __init__(self, device='cpu', weight=1):
        super(Loss4BidirFlowVec_OcclusionAware, self).__init__()
        self.device = device
        self.weight = weight
        self.epsilon = 0.001
    def forward(self, UV_AtoB, UV_BtoA, validMask=torch.tensor([])):
        N, C, H, W = UV_AtoB.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(N,1,1,1)
        yy = yy.view(1,1,H,W).repeat(N,1,1,1)
        GridXY = torch.cat((xx,yy),1).float()
        GridXY = GridXY.to(self.device)
        GridXY = Variable(GridXY)
        #GridXY[:,0,:,:] = 2.0*GridXY[:,0,:,:]/max(W-1,1)-1.0
        #GridXY[:,1,:,:] = 2.0*GridXY[:,1,:,:]/max(H-1,1)-1.0
        GridXY[:,0,:,:] = GridXY[:,0,:,:]/max(W-1,1)
        GridXY[:,1,:,:] = GridXY[:,1,:,:]/max(H-1,1)
        
        mapXY_ABA = self.funcOpticalFlowWarp(GridXY, UV_AtoB.to(self.device), self.device)
        mapXY_ABA = self.funcOpticalFlowWarp(mapXY_ABA, UV_BtoA.to(self.device), self.device)
        self.DiffABA = GridXY-mapXY_ABA
        self.DiffABA = torch.sqrt(torch.sum(self.DiffABA**2, 1)+self.epsilon**2)
        
        totloss_ABA = torch.sum(torch.sum(self.DiffABA,2),1)/(H*W)
        totloss_ABA = torch.sum(torch.sum(totloss_ABA))/N
        

        mapXY_BAB = self.funcOpticalFlowWarp(GridXY, UV_BtoA.to(self.device), self.device)
        mapXY_BAB = self.funcOpticalFlowWarp(mapXY_BAB, UV_AtoB.to(self.device), self.device)
        self.DiffBAB = GridXY-mapXY_BAB  
        self.DiffBAB = torch.sqrt(torch.sum(self.DiffBAB**2, 1)+self.epsilon**2)
        
        totloss_BAB = torch.sum(torch.sum(self.DiffBAB,2),1)/(H*W)
        totloss_BAB = torch.sum(torch.sum(totloss_BAB))/N
        
        return (totloss_BAB+totloss_ABA)*self.weight
   

    def funcOpticalFlowWarp(self, x, flo, device='cpu'):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        grid = grid.to(device)
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
        mask = F.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        return output*mask
    
    
    
class Loss4BidirFlowVecWithoutBorder(nn.Module):
    def __init__(self, device='cpu', weight=1, borderPixel=3):
        super(Loss4BidirFlowVecWithoutBorder, self).__init__()
        self.device = device
        self.weight = weight
        self.borderPixel = borderPixel
        self.epsilon = 0.001
    def forward(self, UV_AtoB, UV_BtoA, borderPixel=-1):
        N, C, H, W = UV_AtoB.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(N,1,1,1)
        yy = yy.view(1,1,H,W).repeat(N,1,1,1)
        GridXY = torch.cat((xx,yy),1).float()
        GridXY = GridXY.to(self.device)
        GridXY = Variable(GridXY)
        #GridXY[:,0,:,:] = 2.0*GridXY[:,0,:,:]/max(W-1,1)-1.0
        #GridXY[:,1,:,:] = 2.0*GridXY[:,1,:,:]/max(H-1,1)-1.0
        GridXY[:,0,:,:] = GridXY[:,0,:,:]/max(W-1,1)
        GridXY[:,1,:,:] = GridXY[:,1,:,:]/max(H-1,1)
        

        
        if borderPixel>=0: self.borderPixel = borderPixel
        self.validMask = torch.ones(N,H,W)
        if self.borderPixel>0:
            self.validMask[:,:self.borderPixel,:] = 0
            self.validMask[:,-self.borderPixel:,:] = 0            
            self.validMask[:,:, :self.borderPixel] = 0
            self.validMask[:,:, -self.borderPixel:] = 0
        self.validMask = Variable(self.validMask).to(self.device)
        
        
        mapXY_ABA = self.funcOpticalFlowWarp(GridXY, UV_AtoB.to(self.device), self.device)
        mapXY_ABA = self.funcOpticalFlowWarp(mapXY_ABA, UV_BtoA.to(self.device), self.device)
        self.DiffABA = GridXY-mapXY_ABA
        #self.DiffABA = torch.sqrt(torch.sum(self.DiffABA**2, 1))#.squeeze()
        self.DiffABA = torch.sqrt(torch.sum(self.DiffABA**2, 1)+self.epsilon**2)        
        self.DiffABA = self.DiffABA*self.validMask
        totloss_ABA = torch.sum(torch.sum(self.DiffABA,2),1)/(H*W)
        totloss_ABA = torch.sum(torch.sum(totloss_ABA))/N
        

        mapXY_BAB = self.funcOpticalFlowWarp(GridXY, UV_BtoA.to(self.device), self.device)
        mapXY_BAB = self.funcOpticalFlowWarp(mapXY_BAB, UV_AtoB.to(self.device), self.device)
        self.DiffBAB = GridXY-mapXY_BAB
        #self.DiffBAB = torch.sqrt(torch.sum(self.DiffBAB**2, 1))#.squeeze()        
        self.DiffBAB = torch.sqrt(torch.sum(self.DiffBAB**2, 1)+self.epsilon**2)        
        self.DiffBAB = self.DiffBAB*self.validMask
        
        
        totloss_BAB = torch.sum(torch.sum(self.DiffBAB,2),1)/(H*W)
        totloss_BAB = torch.sum(torch.sum(totloss_BAB))/N
        
        return (totloss_BAB+totloss_ABA)*self.weight
   
    def funcOpticalFlowWarp(self, x, flo, device='cpu'):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        grid = grid.to(device)
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
        mask = F.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        return output*mask    