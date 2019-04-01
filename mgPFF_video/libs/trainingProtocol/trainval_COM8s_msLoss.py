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
                loss_warp4reconstruction,
                loss_filterSmoothness,
                loss_imageGradient,
                optimizer, scheduler, 
                num_epochs=25, work_dir='./', device='cpu',
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
            print(phase)
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
                imgListA8, imgListB8, imgListA16, imgListB16, imgListA32, imgListB32 = sample
                imgListA32 = imgListA32.to(device)
                imgListB32 = imgListB32.to(device)
                imgListA16 = imgListA16.to(device)
                imgListB16 = imgListB16.to(device)
                imgListA8 = imgListA8.to(device)
                imgListB8 = imgListB8.to(device)
                N = imgListA32.size(0)
                
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                loss32s = 0
                loss16s = 0
                loss8s = 0                
                loss = 0
                with torch.set_grad_enabled(phase=='train'):
                    if phase=='train':  model.train()  # backward + optimize only if in training phase                        
                    else: model.eval()
                    
                    
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
                    
                    
                    
                    recImgA32x4 = warpImgWithScale4(imgListB8, PFFx32_2to1)                                        
                    recImgA16x2 = warpImgWithScale2(recImgA32x4, PFFx16_2to1)                 
                    recImgA16x2 = recImgA16x2.detach()                     
                    recImgB32x4 = warpImgWithScale4(imgListA8, PFFx32_1to2)                                        
                    recImgB16x2 = warpImgWithScale2(recImgB32x4, PFFx16_1to2)                 
                    recImgB16x2 = recImgB16x2.detach() 
                                        
                    #########  at the scale of current interest  #########
                    embFeature2_to_1, embFeature1_to_2 = model(recImgA16x2, imgListA8) 
                    lossRecA = loss_pixelReconstruction(recImgA16x2, imgListA8, embFeature1_to_2)
                    reconsturctedImageA = loss_pixelReconstruction.reconstructImage
                    loss += lossRecA
                    running_loss_reconstruction += lossRecA.item()*N                    
                    
                    lossFlow4ReconA = loss_warp4reconstruction(recImgA16x2, imgListA8, embFeature1_to_2)
                    loss += lossFlow4ReconA
                    running_loss_flow4warpRecon += lossFlow4ReconA.item()*N
                    
                    lossSmooth2to1 = loss_filterSmoothness(embFeature2_to_1)
                    lossSmooth1to2 = loss_filterSmoothness(embFeature1_to_2)
                    loss += lossSmooth2to1
                    loss += lossSmooth1to2
                    running_loss_filterSmoothness += lossSmooth2to1.item()*N
                    running_loss_filterSmoothness += lossSmooth1to2.item()*N                         
                                       
                    loss_imageGradientA = weight4ImGrad*loss_imageGradient(
                        reconsturctedImageA, imgListA8)*N
                    loss += loss_imageGradientA                    
                    running_loss_imageGradient += loss_imageGradientA.item()*N
                    
                    
                    
                    
                    embFeature2_to_1, embFeature1_to_2 = model(recImgB16x2, imgListB8) 
                    lossRecB = loss_pixelReconstruction(recImgB16x2, imgListB8, embFeature1_to_2)
                    reconsturctedImageB = loss_pixelReconstruction.reconstructImage
                    loss += lossRecB
                    running_loss_reconstruction += lossRecB.item()*N
                    
                    lossFlow4ReconB = loss_warp4reconstruction(recImgB16x2, imgListB8, embFeature1_to_2)
                    loss += lossFlow4ReconB
                    running_loss_flow4warpRecon += lossFlow4ReconB.item()*N
                    
                    lossSmooth2to1 = loss_filterSmoothness(embFeature2_to_1)
                    lossSmooth1to2 = loss_filterSmoothness(embFeature1_to_2)
                    loss += lossSmooth2to1
                    loss += lossSmooth1to2
                    running_loss_filterSmoothness += lossSmooth2to1.item()*N
                    running_loss_filterSmoothness += lossSmooth1to2.item()*N                          
                                       
                    loss_imageGradientB = weight4ImGrad*loss_imageGradient(
                        reconsturctedImageB, imgListB8)*N
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
