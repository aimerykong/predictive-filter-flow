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


def train_model(model, dataloaders, dataset_sizes, 
                loss_pixelReconstruction,
                warpImgWithScale1, warpImgWithScale2, warpImgWithScale4,
                warpImgWithScale8, warpImgWithScale16,
                loss_warp4reconstruction,
                loss_filterSmoothness,
                loss_imageGradient,
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
                    loss_warp4reconstruction.device = device
                    loss_filterSmoothness.device = device
                    loss_imageGradient.device = device
                    ################## scale 1/32 ##################
                    loss_filterSmoothness.maxRangePixel = 3
                    
                    PFFx32_2to1, PFFx32_1to2 = model(imgListA32, imgListB32)
                    lossRecB32 = loss_pixelReconstruction(imgListA32, imgListB32, PFFx32_1to2)
                    loss32s += lossRecB32
                    lossRecA32 = loss_pixelReconstruction(imgListB32, imgListA32, PFFx32_2to1)
                    loss32s += lossRecA32
                    lossFlow4ReconB = loss_warp4reconstruction(imgListA32, imgListB32, PFFx32_1to2) 
                    loss32s += lossFlow4ReconB
                    lossFlow4ReconA = loss_warp4reconstruction(imgListB32, imgListA32, PFFx32_2to1)
                    loss32s += lossFlow4ReconA
                    lossSmooth2to1 = loss_filterSmoothness(PFFx32_2to1)
                    lossSmooth1to2 = loss_filterSmoothness(PFFx32_1to2)
                    loss32s += lossSmooth2to1
                    loss32s += lossSmooth1to2
                    
                    if phase=='train': 
                        loss32s.backward()
                    
                    recImgA32x2 = warpImgWithScale2(imgListB16, PFFx32_2to1)
                    recImgA32x2 = recImgA32x2.detach()
                    recImgB32x2 = warpImgWithScale2(imgListA16, PFFx32_1to2)
                    recImgB32x2 = recImgB32x2.detach()
                    
                    recImgA32x4 = warpImgWithScale4(imgListB8, PFFx32_2to1)
                    recImgA32x4 = recImgA32x4.detach()
                    recImgB32x4 = warpImgWithScale4(imgListA8, PFFx32_1to2)
                    recImgB32x4 = recImgB32x4.detach()
                    
                    recImgA32x8 = warpImgWithScale8(imgListB4, PFFx32_2to1)
                    recImgA32x8 = recImgA32x8.detach()
                    recImgB32x8 = warpImgWithScale8(imgListA4, PFFx32_1to2)
                    recImgB32x8 = recImgB32x8.detach()
                    
                    recImgA32x16 = warpImgWithScale16(imgListB2, PFFx32_2to1)
                    recImgA32x16 = recImgA32x16.detach()
                    recImgB32x16 = warpImgWithScale16(imgListA2, PFFx32_1to2)
                    recImgB32x16 = recImgB32x16.detach()
                    
                    
                    ################## scale 1/16 ##################
                    loss_filterSmoothness.maxRangePixel += 1
                    
                    PFFx16_2to1, _ = model(imgListA16, recImgA32x2)
                    _, PFFx16_1to2 = model(recImgB32x2, imgListB16)
                    
                    lossRecA16 = loss_pixelReconstruction(recImgA32x2, imgListA16, PFFx16_2to1)
                    loss16s += lossRecA16
                    lossRecB16 = loss_pixelReconstruction(recImgB32x2, imgListB16, PFFx16_1to2)
                    loss16s += lossRecB16
                    lossFlow4ReconA = loss_warp4reconstruction(recImgA32x2, imgListA16, PFFx16_2to1) 
                    loss16s += lossFlow4ReconA
                    lossFlow4ReconB = loss_warp4reconstruction(recImgB32x2, imgListB16, PFFx16_1to2)
                    loss16s += lossFlow4ReconB
                    lossSmooth2to1 = loss_filterSmoothness(PFFx16_2to1)
                    lossSmooth1to2 = loss_filterSmoothness(PFFx16_1to2)
                    loss16s += lossSmooth2to1
                    loss16s += lossSmooth1to2
                    
                    if phase=='train': 
                        loss16s.backward()                    
                    
                    recImgA16x2 = warpImgWithScale2(recImgA32x4, PFFx16_2to1)                 
                    recImgA16x2 = recImgA16x2.detach()                     
                    recImgB16x2 = warpImgWithScale2(recImgB32x4, PFFx16_1to2)                 
                    recImgB16x2 = recImgB16x2.detach() 
                    
                    recImgA16x4 = warpImgWithScale4(recImgA32x8, PFFx16_2to1)                 
                    recImgA16x4 = recImgA16x4.detach()                     
                    recImgB16x4 = warpImgWithScale4(recImgB32x8, PFFx16_1to2)                 
                    recImgB16x4 = recImgB16x4.detach() 
                    
                    recImgA16x8 = warpImgWithScale8(recImgA32x16, PFFx16_2to1)                 
                    recImgA16x8 = recImgA16x8.detach()                     
                    recImgB16x8 = warpImgWithScale8(recImgB32x16, PFFx16_1to2)                 
                    recImgB16x8 = recImgB16x8.detach() 
                    
                    
                    
                    ################## scale 1/8 ##################
                    loss_filterSmoothness.maxRangePixel += 1
                    
                    PFFx8_2to1, _ = model(imgListA8, recImgA16x2)
                    _, PFFx8_1to2 = model(recImgB16x2, imgListB8)
                    
                    lossRecA8 = loss_pixelReconstruction(recImgA16x2, imgListA8, PFFx8_2to1)
                    loss8s += lossRecA8
                    lossRecB8 = loss_pixelReconstruction(recImgB16x2, imgListB8, PFFx8_1to2)
                    loss8s += lossRecB8
                    lossFlow4ReconA = loss_warp4reconstruction(recImgA16x2, imgListA8, PFFx8_2to1) 
                    loss8s += lossFlow4ReconA
                    lossFlow4ReconB = loss_warp4reconstruction(recImgB16x2, imgListB8, PFFx8_1to2)
                    loss8s += lossFlow4ReconB
                    lossSmooth2to1 = loss_filterSmoothness(PFFx8_2to1)
                    lossSmooth1to2 = loss_filterSmoothness(PFFx8_1to2)
                    loss8s += lossSmooth2to1
                    loss8s += lossSmooth1to2
                    
                    if phase=='train': 
                        loss8s.backward()   
                    
                    recImgA8x2 = warpImgWithScale2(recImgA16x4, PFFx8_2to1)                 
                    recImgA8x2 = recImgA8x2.detach()                     
                    recImgB8x2 = warpImgWithScale2(recImgB16x4, PFFx8_1to2)                 
                    recImgB8x2 = recImgB8x2.detach()                     
                        
                    recImgA8x4 = warpImgWithScale4(recImgA16x8, PFFx8_2to1)                 
                    recImgA8x4 = recImgA8x4.detach()                     
                    recImgB8x4 = warpImgWithScale4(recImgB16x8, PFFx8_1to2)                 
                    recImgB8x4 = recImgB8x4.detach()  
                        
                        
                        
                        
                        
                    ################## scale 1/4 ##################
                    loss_filterSmoothness.maxRangePixel += 1
                    
                    PFFx4_2to1, _ = model(imgListA4, recImgA8x2)
                    _, PFFx4_1to2 = model(recImgB8x2, imgListB4)
                    
                    lossRecA4 = loss_pixelReconstruction(recImgA8x2, imgListA4, PFFx4_2to1)
                    loss4s += lossRecA4
                    lossRecB4 = loss_pixelReconstruction(recImgB8x2, imgListB4, PFFx4_1to2)
                    loss4s += lossRecB4
                    lossFlow4ReconA = loss_warp4reconstruction(recImgA8x2, imgListA4, PFFx4_2to1) 
                    loss4s += lossFlow4ReconA
                    lossFlow4ReconB = loss_warp4reconstruction(recImgB8x2, imgListB4, PFFx4_1to2)
                    loss4s += lossFlow4ReconB
                    lossSmooth2to1 = loss_filterSmoothness(PFFx4_2to1)
                    lossSmooth1to2 = loss_filterSmoothness(PFFx4_1to2)
                    loss4s += lossSmooth2to1
                    loss4s += lossSmooth1to2
                    
                    if phase=='train': 
                        loss4s.backward()   
                    
                    recImgA4x2 = warpImgWithScale2(recImgA8x4, PFFx4_2to1)                 
                    recImgA4x2 = recImgA4x2.detach()
                    recImgB4x2 = warpImgWithScale2(recImgB8x4, PFFx4_1to2)                 
                    recImgB4x2 = recImgB4x2.detach()
                    
                    
                    
                    ################## scale 1/2 ##################
                    model.to(supplDevice)
                    recImgA4x2 = recImgA4x2.to(supplDevice)
                    imgListA2 = imgListA2.to(supplDevice)
                    recImgB4x2 = recImgB4x2.to(supplDevice)
                    imgListB2 = imgListB2.to(supplDevice)                    
                    loss_filterSmoothness.maxRangePixel += 1
                    loss_warp4reconstruction.device = supplDevice
                    loss_filterSmoothness.device = supplDevice
                    loss_imageGradient.device = supplDevice
                    
                    #####  at the scale of current interest  ######
                    _, PFFx2A_1to2 = model(recImgA4x2, imgListA2) 
                    lossRecA = loss_pixelReconstruction(recImgA4x2, imgListA2, PFFx2A_1to2)
                    loss2s += lossRecA
                    running_loss_reconstruction += lossRecA.item()*N                    
                    
                    lossFlow4ReconA = loss_warp4reconstruction(recImgA4x2, imgListA2, PFFx2A_1to2)
                    reconsturctedImageA = loss_warp4reconstruction.reconstructImage
                    loss2s += lossFlow4ReconA
                    running_loss_flow4warpRecon += lossFlow4ReconA.item()*N
                    
                    #lossSmooth2to1A = loss_filterSmoothness(PFFx2A_2to1)
                    lossSmooth1to2A = loss_filterSmoothness(PFFx2A_1to2)
                    #loss += lossSmooth2to1A
                    loss2s += lossSmooth1to2A
                    #running_loss_filterSmoothness += lossSmooth2to1A.item()*N
                    running_loss_filterSmoothness += lossSmooth1to2A.item()*N                         
                                       
                    loss_imageGradientA = loss_imageGradient(reconsturctedImageA, imgListA2)*N
                    loss2s += loss_imageGradientA                    
                    running_loss_imageGradient += loss_imageGradientA.item()*N
                    
                    
                    
                    
                    _, PFFx2B_1to2 = model(recImgB4x2, imgListB2) 
                    lossRecB = loss_pixelReconstruction(recImgB4x2, imgListB2, PFFx2B_1to2)
                    loss2s += lossRecB
                    running_loss_reconstruction += lossRecB.item()*N
                    
                    lossFlow4ReconB = loss_warp4reconstruction(recImgB4x2, imgListB2, PFFx2B_1to2)
                    reconsturctedImageB = loss_warp4reconstruction.reconstructImage
                    loss2s += lossFlow4ReconB
                    running_loss_flow4warpRecon += lossFlow4ReconB.item()*N
                    
                    #lossSmooth2to1B = loss_filterSmoothness(PFFx2B_2to1)
                    lossSmooth1to2B = loss_filterSmoothness(PFFx2B_1to2)
                    #loss += lossSmooth2to1B
                    loss2s += lossSmooth1to2B
                    #running_loss_filterSmoothness += lossSmooth2to1B.item()*N
                    running_loss_filterSmoothness += lossSmooth1to2B.item()*N                          
                                       
                    loss_imageGradientB = loss_imageGradient(reconsturctedImageB, imgListB2)*N
                    loss2s += loss_imageGradientB                    
                    running_loss_imageGradient += loss_imageGradientB.item()*N
                        
                    if phase=='train': 
                        loss2s.backward()
                        optimizer.step()
                    
                    loss = loss2s
                    
                    
                    
                    
                    
                    
                    
                # statistics  
                iterCount += 1
                sampleCount += N                                
                running_loss += loss.item() * N                             
                print2screen_avgLoss = running_loss/sampleCount
                print2screen_avgLoss_Rec = running_loss_reconstruction/sampleCount
                print2screen_avgLoss_Smooth = running_loss_filterSmoothness/sampleCount
                print2screen_avgLoss_imgGrad = running_loss_imageGradient/sampleCount
                print2screen_avgLoss_flow4warpRecon = running_loss_flow4warpRecon/sampleCount
                
                       
                del loss
                if iterCount%100==0:
                    print('\t{}/{} loss: {:.4f} Rec:{:.3f}, FVrec:{:.3f}, Smooth:{:.3f}, imGrad:{:.3f}'.
                          format(
                              iterCount, 
                              len(dataloaders[phase]), 
                              print2screen_avgLoss, 
                              print2screen_avgLoss_Rec,
                              print2screen_avgLoss_flow4warpRecon,
                              print2screen_avgLoss_Smooth,
                              print2screen_avgLoss_imgGrad)                          
                         )
                    fn = open(log_filename,'a')        
                    fn.write('\t{}/{} loss: {:.4f} Rec:{:.3f}, FVrec:{:.3f}, Smooth:{:.3f}, imGrad:{:.3f}\n'.
                             format(
                                 iterCount, 
                                 len(dataloaders[phase]), 
                                 print2screen_avgLoss, 
                                 print2screen_avgLoss_Rec,
                                 print2screen_avgLoss_flow4warpRecon,
                                 print2screen_avgLoss_Smooth,
                                 print2screen_avgLoss_imgGrad)
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
                fn.write('\t{:.4f} Rec:{:.3f}, FVrec:{:.3f}, Smooth:{:.3f}, imGrad:{:.3f}\n'.
                         format(
                             print2screen_avgLoss, 
                             print2screen_avgLoss_Rec,
                             print2screen_avgLoss_flow4warpRecon,
                             print2screen_avgLoss_Smooth,
                             print2screen_avgLoss_imgGrad)
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
