import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
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

from utils.metrics import *

def genFlowVector4Visualization(F_fine2coarse):
    F1tmp = F_fine2coarse[0]
    H,W = F1tmp.shape[1:]

    maxvalXMask = torch.ones(1,1)*(W-1)
    maxvalXMask = maxvalXMask.repeat(H,W)#.to(device)
    maxvalYMask = torch.ones(1,1)*(H-1)
    maxvalYMask = maxvalYMask.repeat(H,W)#.to(device)
    minvalMask = torch.zeros(1,1)
    minvalMask = minvalMask.repeat(H,W)#.to(device)


    UV = torch.zeros_like(F1tmp)
    grid_x = torch.arange(0, W).view(1,-1).repeat(H,1).float() #.to(device)
    grid_y = torch.arange(0, H).view(-1,1).repeat(1,W).float() #.to(device)
    #ylist, xlist = grid_y.numpy(), grid_x.numpy()
    ycoord, xcoord = grid_y, grid_x
    for i, Fvec in enumerate(F_fine2coarse):    
        xcoord_round = torch.round(xcoord)
        xcoord_round = clipTensor(xcoord_round,maxvalXMask,minvalMask)
        ycoord_round = torch.round(ycoord)
        ycoord_round = clipTensor(ycoord_round,maxvalYMask,minvalMask)
        xcoord_ceil = torch.ceil(xcoord)
        xcoord_ceil = clipTensor(xcoord_ceil,maxvalXMask,minvalMask)
        xcoord_floor = torch.floor(xcoord)
        xcoord_floor = clipTensor(xcoord_floor,maxvalXMask,minvalMask)
        ycoord_ceil = torch.ceil(ycoord)
        ycoord_ceil = clipTensor(ycoord_ceil,maxvalYMask,minvalMask)
        ycoord_floor = torch.floor(ycoord)
        ycoord_floor = clipTensor(ycoord_floor,maxvalYMask,minvalMask)


        xcoord_round = xcoord_round.detach().cpu().numpy()
        ycoord_round = ycoord_round.detach().cpu().numpy()      
        xcoord_ceil = xcoord_ceil.detach().cpu().numpy()
        xcoord_floor = xcoord_floor.detach().cpu().numpy()
        ycoord_ceil = ycoord_ceil.detach().cpu().numpy()
        ycoord_floor = ycoord_floor.detach().cpu().numpy()        


        xlist_supp_round, ylist_supp_round = Fvec[0,ycoord_round,xcoord_round], Fvec[1,ycoord_round,xcoord_round] 
        xlist_supp_UL, ylist_supp_UL = Fvec[0,ycoord_floor,xcoord_floor], Fvec[1,ycoord_floor,xcoord_floor] 
        xlist_supp_UR, ylist_supp_UR = Fvec[0,ycoord_floor,xcoord_ceil], Fvec[1,ycoord_floor,xcoord_ceil] 
        xlist_supp_BL, ylist_supp_BL = Fvec[0,ycoord_ceil,xcoord_floor], Fvec[1,ycoord_ceil,xcoord_floor] 
        xlist_supp_BR, ylist_supp_BR = Fvec[0,ycoord_ceil,xcoord_ceil], Fvec[1,ycoord_ceil,xcoord_ceil] 

        xcoord_ceil = torch.from_numpy(xcoord_ceil)
        xcoord_floor = torch.from_numpy(xcoord_floor)
        ycoord_ceil = torch.from_numpy(ycoord_ceil)
        ycoord_floor = torch.from_numpy(ycoord_floor)


        dominatorTMP = xcoord_ceil-xcoord_floor
        dominatorTMP[dominatorTMP==0]=1
        wLeft = xcoord_ceil-xcoord
        wRight = xcoord-xcoord_floor
        wLeft[wLeft==0]=0.5
        wRight[wRight==0]=0.5

        xlist_supp_u = wLeft/dominatorTMP*xlist_supp_UL + wRight/dominatorTMP*xlist_supp_UR 
        xlist_supp_b = wLeft/dominatorTMP*xlist_supp_BL + wRight/dominatorTMP*xlist_supp_BR 

        dominatorTMP = ycoord_ceil-ycoord_floor
        dominatorTMP[dominatorTMP==0]=1
        wUpper = ycoord_ceil-ycoord
        wBottom = ycoord-ycoord_floor
        wUpper[wUpper==0]=0.5
        wBottom[wBottom==0]=0.5
        xlist_supp =  wUpper/dominatorTMP*xlist_supp_u + wBottom/dominatorTMP*xlist_supp_b


        dominatorTMP = xcoord_ceil-xcoord_floor
        dominatorTMP[dominatorTMP==0]=1
        wLeft = xcoord_ceil-xcoord
        wRight = xcoord-xcoord_floor
        wLeft[wLeft==0]=0.5
        wRight[wRight==0]=0.5

        ylist_supp_u = wLeft/dominatorTMP*ylist_supp_UL + wRight/dominatorTMP*ylist_supp_UR 
        ylist_supp_b = wLeft/dominatorTMP*ylist_supp_BL + wRight/dominatorTMP*ylist_supp_BR 

        dominatorTMP = ycoord_ceil-ycoord_floor
        dominatorTMP[dominatorTMP==0]=1
        wUpper = ycoord_ceil-ycoord
        wBottom = ycoord-ycoord_floor
        wUpper[wUpper==0]=0.5
        wBottom[wBottom==0]=0.5
        ylist_supp =  wUpper/dominatorTMP*ylist_supp_u + wBottom/dominatorTMP*ylist_supp_b


        if i==len(F_fine2coarse)-1:
            xlist_supp, ylist_supp = xlist_supp_round, ylist_supp_round
            #xlist, ylist = xcoord-grid_x.detach().cpu(), ycoord-grid_y.detach().cpu()
            #xlist, ylist = torch.round(xlist), torch.round(ylist)

        xcoord, ycoord = xlist_supp+xcoord, ylist_supp+ycoord
        xcoord = xcoord.detach().cpu()#.numpy()
        ycoord = ycoord.detach().cpu()#.numpy()
        
        if i==len(F_fine2coarse)-1:
            xlist, ylist = xcoord-grid_x.detach().cpu(), ycoord-grid_y.detach().cpu()

    UV[0] = xlist.view(1,H,W)
    UV[1] = ylist.view(1,H,W)
    return UV




def funcOpticalFlowWarp(x, flo, device='cpu'):
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
    vgrid = Variable(grid).to(device) + flo.to(device)

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
    mask = F.grid_sample(mask.to(device), vgrid.to(device))

    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    return output*mask, mask




def train_model(model, dataloaders, dataset_sizes, 
                loss_pixelReconstruction,
                warpImgWithScale1, warpImgWithScale2, warpImgWithScale4,
                warpImgWithScale8, warpImgWithScale16,
                warpImgWithUV,
                loss_warp4reconstruction,
                loss_filterSmoothness,
                loss_imageGradient,
                loss_laziness,
                optimizer, scheduler, 
                num_epochs=25, work_dir='./', 
                device='cpu', supplDevice='cpu',
                weight4ImRecon=1,
                weight4ImGrad=1):
    
    log_filename = os.path.join(work_dir,'train.log')    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
        
    for epoch in range(num_epochs):        
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        fn = open(log_filename,'a')
        fn.write('\nEpoch {}/{}\n'.format(epoch+1, num_epochs))
        fn.write('--'*5+'\n')
        fn.close()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            fn = open(log_filename,'a')        
            fn.write(phase+'\n')
            fn.close()
            
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else: model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_loss_reconstruction = 0.0
            running_loss_flow4warpRecon = 0.0
            running_loss_filterSmoothness = 0.0
            running_loss_imageGradient = 0.0
            
            running_loss_laziness_32 = 0.0
            
            # Iterate over data.
            iterCount,sampleCount = 0, 0
            for sample in dataloaders[phase]:
                
                imgListA2,imgListB2, imgListA4,imgListB4, imgListA8,imgListB8 = sample[:6]
                imgListA16,imgListB16,imgListA32,imgListB32 = sample[6:]              
                imgListA32 = imgListA32.to(device)
                imgListB32 = imgListB32.to(device)
                imgListA16 = imgListA16.to(device)
                imgListB16 = imgListB16.to(device)
                imgListA8 = imgListA8.to(device)
                imgListB8 = imgListB8.to(device)
                imgListA4 = imgListA4.to(device)
                imgListB4 = imgListB4.to(device)
                imgListA2 = imgListA2.to(device)
                imgListB2 = imgListB2.to(device)
                N = imgListA32.size(0)
                
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                loss32s = 0
                loss16s = 0
                loss8s = 0  
                loss4s = 0  
                loss2s = 0                
                loss = 0
                with torch.set_grad_enabled(phase=='train'):
                    if phase=='train':  model.train()  # backward + optimize only if in training phase                        
                    else: model.eval()
                    
                    
                    model.to(device)
                    #loss_warp4reconstruction.device = device
                    #loss_filterSmoothness.device = device
                    #loss_imageGradient.device = device
                    
                    ################## scale 1/32 ##################              
                    loss = 0
                    loss_filterSmoothness.maxRangePixel = 3
                    
                    PFFx32_2to1, PFFx32_1to2 = model(imgListA32, imgListB32)
                    lossRecB32 = loss_pixelReconstruction(imgListA32, imgListB32, PFFx32_1to2)
                    loss += lossRecB32
                    lossRecA32 = loss_pixelReconstruction(imgListB32, imgListA32, PFFx32_2to1)
                    loss += lossRecA32
                    lossFlow4ReconB = loss_warp4reconstruction(imgListA32, imgListB32, PFFx32_1to2) 
                    UVx32_1to2 = loss_warp4reconstruction.UVgrid
                    lossLaziness_1to2 = loss_laziness(UVx32_1to2)
                    loss += lossLaziness_1to2
                    running_loss_laziness_32 += lossLaziness_1to2.item()*N
                    loss += lossFlow4ReconB
                    lossFlow4ReconA = loss_warp4reconstruction(imgListB32, imgListA32, PFFx32_2to1)
                    UVx32_2to1 = loss_warp4reconstruction.UVgrid                    
                    lossLaziness_2to1 = loss_laziness(UVx32_2to1)
                    loss += lossLaziness_2to1
                    running_loss_laziness_32 += lossLaziness_2to1.item()*N
                    loss += lossFlow4ReconA
                    lossSmooth2to1 = loss_filterSmoothness(PFFx32_2to1)
                    lossSmooth1to2 = loss_filterSmoothness(PFFx32_1to2)
                    loss += lossSmooth2to1
                    loss += lossSmooth1to2
                                        
                                        
                    if phase=='train': 
                        loss.backward()
                    PFFx32_2to1 = PFFx32_2to1.detach()
                    PFFx32_1to2 = PFFx32_1to2.detach()
                    UVx32_1to2 = UVx32_1to2.detach().type(torch.LongTensor).type(torch.FloatTensor).to(device)
                    UVx32_2to1 = UVx32_2to1.detach().type(torch.LongTensor).type(torch.FloatTensor).to(device)
                    
                    
                    
                    recImgA32x2 = warpImgWithUV(imgListB16, UVx32_2to1, 2).detach()
                    recImgB32x2 = warpImgWithUV(imgListA16, UVx32_1to2, 2).detach()                    
                    recImgA32x4 = warpImgWithUV(imgListB8, UVx32_2to1, 4).detach()
                    recImgB32x4 = warpImgWithUV(imgListA8, UVx32_1to2, 4).detach()                    
                    recImgA32x8 = warpImgWithUV(imgListB4, UVx32_2to1, 8).detach()
                    recImgB32x8 = warpImgWithUV(imgListA4, UVx32_1to2, 8).detach()                    
                    recImgA32x16 = warpImgWithUV(imgListB2, UVx32_2to1, 16).detach()
                    recImgB32x16 = warpImgWithUV(imgListA2, UVx32_1to2, 16).detach()
                   
                    
                    ################## scale 1/16 ##################         
                    loss = 0
                    loss_filterSmoothness.maxRangePixel += 1
                    
                    PFFx16_2to1, _ = model(imgListA16, recImgA32x2)
                    _, PFFx16_1to2 = model(recImgB32x2, imgListB16)
                    
                    lossRecA16 = loss_pixelReconstruction(recImgA32x2, imgListA16, PFFx16_2to1)
                    loss += lossRecA16
                    lossRecB16 = loss_pixelReconstruction(recImgB32x2, imgListB16, PFFx16_1to2)
                    loss += lossRecB16
                    lossFlow4ReconA = loss_warp4reconstruction(recImgA32x2, imgListA16, PFFx16_2to1) 
                    UVx16_2to1 = loss_warp4reconstruction.UVgrid.detach().type(
                        torch.LongTensor).type(torch.FloatTensor).to(device)
                    loss += lossFlow4ReconA
                    lossFlow4ReconB = loss_warp4reconstruction(recImgB32x2, imgListB16, PFFx16_1to2)
                    UVx16_1to2 = loss_warp4reconstruction.UVgrid.detach().type(
                        torch.LongTensor).type(torch.FloatTensor).to(device)
                    loss += lossFlow4ReconB
                    lossSmooth2to1 = loss_filterSmoothness(PFFx16_2to1)
                    lossSmooth1to2 = loss_filterSmoothness(PFFx16_1to2)
                    loss += lossSmooth2to1
                    loss += lossSmooth1to2
                    
                    if phase=='train': 
                        loss.backward()       
                    PFFx16_2to1 = PFFx16_2to1.detach()
                    PFFx16_1to2 = PFFx16_1to2.detach()    
                    
                    recImgA16x2 = warpImgWithUV(recImgA32x4, UVx16_2to1, 2).detach()
                    recImgB16x2 = warpImgWithUV(recImgB32x4, UVx16_1to2, 2).detach()   
                    recImgA16x4 = warpImgWithUV(recImgA32x8, UVx16_2to1, 4).detach()
                    recImgB16x4 = warpImgWithUV(recImgB32x8, UVx16_1to2, 4).detach()   
                    recImgA16x8 = warpImgWithUV(recImgA32x16, UVx16_2to1, 8).detach()
                    recImgB16x8 = warpImgWithUV(recImgB32x16, UVx16_1to2, 8).detach()                                         

                    
                    
                    ################## scale 1/8 ##################         
                    loss = 0
                    loss_filterSmoothness.maxRangePixel += 1
                    
                    PFFx8_2to1, _ = model(imgListA8, recImgA16x2)
                    _, PFFx8_1to2 = model(recImgB16x2, imgListB8)
                    
                    lossRecA8 = loss_pixelReconstruction(recImgA16x2, imgListA8, PFFx8_2to1)
                    loss += lossRecA8
                    lossRecB8 = loss_pixelReconstruction(recImgB16x2, imgListB8, PFFx8_1to2)
                    loss += lossRecB8
                    lossFlow4ReconA = loss_warp4reconstruction(recImgA16x2, imgListA8, PFFx8_2to1) 
                    UVx8_2to1 = loss_warp4reconstruction.UVgrid.detach().type(
                        torch.LongTensor).type(torch.FloatTensor).to(device)
                    loss += lossFlow4ReconA
                    lossFlow4ReconB = loss_warp4reconstruction(recImgB16x2, imgListB8, PFFx8_1to2)
                    UVx8_1to2 = loss_warp4reconstruction.UVgrid.detach().type(
                        torch.LongTensor).type(torch.FloatTensor).to(device)
                    loss += lossFlow4ReconB
                    lossSmooth2to1 = loss_filterSmoothness(PFFx8_2to1)
                    lossSmooth1to2 = loss_filterSmoothness(PFFx8_1to2)
                    loss += lossSmooth2to1
                    loss += lossSmooth1to2
                    
                    if phase=='train': 
                        loss.backward()   
                    PFFx8_2to1 = PFFx8_2to1.detach()
                    PFFx8_1to2 = PFFx8_1to2.detach()   
                    
                    recImgA8x2 = warpImgWithUV(recImgA16x4, UVx8_2to1, 2).detach()
                    recImgB8x2 = warpImgWithUV(recImgB16x4, UVx8_1to2, 2).detach()   
                    recImgA8x4 = warpImgWithUV(recImgA16x8, UVx8_2to1, 4).detach()
                    recImgB8x4 = warpImgWithUV(recImgB16x8, UVx8_1to2, 4).detach() 
                    
                        
                    ################## scale 1/4 ##################         
                    loss = 0
                    loss_filterSmoothness.maxRangePixel += 1
                    
                    PFFx4_2to1, _ = model(imgListA4, recImgA8x2)
                    _, PFFx4_1to2 = model(recImgB8x2, imgListB4)
                    
                    lossRecA4 = loss_pixelReconstruction(recImgA8x2, imgListA4, PFFx4_2to1)
                    loss += lossRecA4
                    lossRecB4 = loss_pixelReconstruction(recImgB8x2, imgListB4, PFFx4_1to2)
                    loss += lossRecB4
                    lossFlow4ReconA = loss_warp4reconstruction(recImgA8x2, imgListA4, PFFx4_2to1) 
                    UVx4_2to1 = loss_warp4reconstruction.UVgrid.detach().type(
                        torch.LongTensor).type(torch.FloatTensor).to(device)
                    loss += lossFlow4ReconA
                    lossFlow4ReconB = loss_warp4reconstruction(recImgB8x2, imgListB4, PFFx4_1to2)
                    UVx4_1to2 = loss_warp4reconstruction.UVgrid.detach().type(
                        torch.LongTensor).type(torch.FloatTensor).to(device)
                    loss += lossFlow4ReconB
                    lossSmooth2to1 = loss_filterSmoothness(PFFx4_2to1)
                    lossSmooth1to2 = loss_filterSmoothness(PFFx4_1to2)
                    loss += lossSmooth2to1
                    loss += lossSmooth1to2
                    
                    if phase=='train': 
                        loss.backward()   
                    PFFx4_2to1 = PFFx4_2to1.detach()
                    PFFx4_1to2 = PFFx4_1to2.detach()   
                                        
                    recImgA4x2 = warpImgWithUV(recImgA8x4, UVx4_2to1, 2).detach()
                    recImgB4x2 = warpImgWithUV(recImgB8x4, UVx4_1to2, 2).detach()                      
   
                    ################## scale 1/2 ##################         
                    loss = 0
                    #model.to(supplDevice)
                    #recImgA4x2 = recImgA4x2.to(supplDevice)
                    #imgListA2 = imgListA2.to(supplDevice)
                    #recImgB4x2 = recImgB4x2.to(supplDevice)
                    #imgListB2 = imgListB2.to(supplDevice)                    
                    loss_filterSmoothness.maxRangePixel += 1
                    #loss_warp4reconstruction.device = supplDevice
                    #loss_filterSmoothness.device = supplDevice
                    #loss_imageGradient.device = supplDevice
                    
                    #####  at the scale of current interest  ######
                    _, PFFx2A_1to2 = model(recImgA4x2, imgListA2) 
                    lossRecA = loss_pixelReconstruction(recImgA4x2, imgListA2, PFFx2A_1to2)
                    loss += lossRecA
                    running_loss_reconstruction += lossRecA.item()*N                    
                    
                    lossFlow4ReconA = loss_warp4reconstruction(recImgA4x2, imgListA2, PFFx2A_1to2)
                    UVx2A_1to2 = loss_warp4reconstruction.UVgrid.detach().type(
                        torch.LongTensor).type(torch.FloatTensor).to(device)
                    reconsturctedImageA = loss_warp4reconstruction.reconstructImage
                    loss += lossFlow4ReconA
                    running_loss_flow4warpRecon += lossFlow4ReconA.item()*N
                    
                    #lossSmooth2to1A = loss_filterSmoothness(PFFx2A_2to1)
                    lossSmooth1to2A = loss_filterSmoothness(PFFx2A_1to2)
                    #loss += lossSmooth2to1A
                    loss += lossSmooth1to2A
                    #running_loss_filterSmoothness += lossSmooth2to1A.item()*N
                    running_loss_filterSmoothness += lossSmooth1to2A.item()*N                         
                                       
                    loss_imageGradientA = loss_imageGradient(reconsturctedImageA, imgListA2)*N
                    loss += loss_imageGradientA                    
                    running_loss_imageGradient += loss_imageGradientA.item()*N
                    
                    
                    
                    
                    _, PFFx2B_1to2 = model(recImgB4x2, imgListB2) 
                    lossRecB = loss_pixelReconstruction(recImgB4x2, imgListB2, PFFx2B_1to2)
                    loss += lossRecB
                    running_loss_reconstruction += lossRecB.item()*N
                    
                    lossFlow4ReconB = loss_warp4reconstruction(recImgB4x2, imgListB2, PFFx2B_1to2)
                    UVx2B_1to2 = loss_warp4reconstruction.UVgrid.detach().type(
                        torch.LongTensor).type(torch.FloatTensor).to(device)
                    reconsturctedImageB = loss_warp4reconstruction.reconstructImage
                    loss += lossFlow4ReconB
                    running_loss_flow4warpRecon += lossFlow4ReconB.item()*N
                    
                    #lossSmooth2to1B = loss_filterSmoothness(PFFx2B_2to1)
                    lossSmooth1to2B = loss_filterSmoothness(PFFx2B_1to2)
                    #loss += lossSmooth2to1B
                    loss += lossSmooth1to2B
                    #running_loss_filterSmoothness += lossSmooth2to1B.item()*N
                    running_loss_filterSmoothness += lossSmooth1to2B.item()*N                          
                                       
                    loss_imageGradientB = loss_imageGradient(reconsturctedImageB, imgListB2)*N
                    loss += loss_imageGradientB                    
                    running_loss_imageGradient += loss_imageGradientB.item()*N
                        
                    if phase=='train': 
                        loss.backward()
                        optimizer.step()
                                        
                    
                    
                # statistics  
                iterCount += 1
                sampleCount += N                                
                running_loss += loss.item() * N                             
                print2screen_avgLoss = running_loss/sampleCount
                print2screen_avgLoss_Rec = running_loss_reconstruction/sampleCount
                print2screen_avgLoss_Smooth = running_loss_filterSmoothness/sampleCount
                print2screen_avgLoss_imgGrad = running_loss_imageGradient/sampleCount
                print2screen_avgLoss_flow4warpRecon = running_loss_flow4warpRecon/sampleCount
                print2screen_laziness = running_loss_laziness_32/sampleCount
                       
                del loss
                if iterCount%100==0:
                    print('\t{}/{} loss: {:.4f} Rec:{:.3f}, FVrec:{:.3f}, Smooth:{:.3f}, imGrad:{:.3f}, lazy:{:3f}'.
                          format(
                              iterCount, 
                              len(dataloaders[phase]), 
                              print2screen_avgLoss, 
                              print2screen_avgLoss_Rec,
                              print2screen_avgLoss_flow4warpRecon,
                              print2screen_avgLoss_Smooth,
                              print2screen_avgLoss_imgGrad,
                              print2screen_laziness)                          
                         )
                    fn = open(log_filename,'a')        
                    fn.write('\t{}/{} loss: {:.4f} Rec:{:.3f}, FVrec:{:.3f}, Smooth:{:.3f}, imGrad:{:.3f}, lazy:{:3f}\n'.
                             format(
                                 iterCount, 
                                 len(dataloaders[phase]), 
                                 print2screen_avgLoss, 
                                 print2screen_avgLoss_Rec,
                                 print2screen_avgLoss_flow4warpRecon,
                                 print2screen_avgLoss_Smooth,
                                 print2screen_avgLoss_imgGrad,
                                 print2screen_laziness)
                            )
                    fn.close()
  
            epoch_loss = running_loss / dataset_sizes[phase]
                    
            print('\tloss: {:.6f}'.format(epoch_loss))
            fn = open(log_filename,'a')
            fn.write('\tloss: {:.6f}\n'.format(epoch_loss))
            fn.close()
                    
                
            # deep copy the model
            cur_model_wts = copy.deepcopy(model.state_dict())
            path_to_save_paramOnly = os.path.join(work_dir, 'epoch-{}.paramOnly'.format(epoch+1))
            torch.save(cur_model_wts, path_to_save_paramOnly)
            
            if phase=='val' and epoch_loss<best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                
                path_to_save_paramOnly = os.path.join(work_dir, 'bestValModel.paramOnly')
                torch.save(best_model_wts, path_to_save_paramOnly)
                #path_to_save_wholeModel = os.path.join(work_dir, 'bestValModel.wholeModel')
                #torch.save(model, path_to_save_wholeModel)
                
                file_to_note_bestModel = os.path.join(work_dir,'note_bestModel.log')
                fn = open(file_to_note_bestModel,'a')
                fn.write('The best model is achieved at epoch-{}: loss{:.6f}.\n'.format(epoch+1,best_loss))
                fn.write('\t{:.4f} Rec:{:.3f}, FVrec:{:.3f}, Smooth:{:.3f}, imGrad:{:.3f}, lazy:{:3f}\n'.
                         format(
                             print2screen_avgLoss, 
                             print2screen_avgLoss_Rec,
                             print2screen_avgLoss_flow4warpRecon,
                             print2screen_avgLoss_Smooth,
                             print2screen_avgLoss_imgGrad,
                             print2screen_laziness)
                        )
                fn.close()
                
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    fn = open(log_filename,'a')
    fn.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    fn.close()
   
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
