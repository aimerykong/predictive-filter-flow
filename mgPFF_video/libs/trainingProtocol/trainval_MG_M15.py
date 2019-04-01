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



def upsamplePFF(PFF, ksize, upF, upFactorX):
    NCHW = PFF.size()                        
    PFF = PFF.view(NCHW[0], ksize, ksize, NCHW[2]*NCHW[3])
    PFF = torch.matmul(upF, PFF)
    PFF = PFF.permute(0,2,1,3)
    PFF = torch.matmul(upF, PFF)
    PFF = PFF.permute(0,2,1,3)
    PFF = PFF.view(NCHW[0], 
                   ksize*upFactorX, ksize*upFactorX, NCHW[2], NCHW[3])
    PFF = PFF.contiguous().view(NCHW[0],
                                ksize*upFactorX*ksize*upFactorX, NCHW[2], NCHW[3])
    PFF = PFF/upFactorX/upFactorX
    return PFF





def train_model(model, Model16, Model8, dataloaders, dataset_sizes, 
                loss_pixelReconstruction, loss_pixelReconstructionX2, loss_pixelReconstructionX4,
                warpImgWithScale1, warpImgWithScale2, warpImgWithScale4,
                loss_warp4reconstruction,
                loss_groupSparsity, 
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
    

    upFactorX2 = 2
    FFsize = ksize = 11    
    upF_x2 = torch.zeros(FFsize*upFactorX2, FFsize)
    for i in range(upFactorX2):
        idxRows = list(range(i,upF_x2.size(0), upFactorX2))
        upF_x2[idxRows, :] = torch.eye(FFsize)
    upF_x2 = upF_x2.to(device)

    upFactorX4 = 4
    FFsize = ksize = 11    
    upF_x4 = torch.zeros(FFsize*upFactorX4, FFsize)
    for i in range(upFactorX4):
        idxRows = list(range(i,upF_x4.size(0), upFactorX4))
        upF_x4[idxRows, :] = torch.eye(FFsize)
    upF_x4 = upF_x4.to(device)    
        
        
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
            running_loss_reconstructionX2 = 0.0
            running_loss_reconstructionX4 = 0.0
            running_loss_flow4warpRecon = 0.0
            running_loss_filterSmoothness = 0.0
            running_loss_groupSparsity = 0.0
            running_loss_imageGradient = 0.0
            running_loss_imageGradientX2 = 0.0
            running_loss_imageGradientX4 = 0.0
            
            
            # Iterate over data.
            iterCount,sampleCount = 0, 0
            for sample in dataloaders[phase]:
                imgListA4,imgListB4,imgListA8,imgListB8,imgListA16,imgListB16=sample
                imgListA4 = imgListA4.to(device)
                imgListB4 = imgListB4.to(device)
                imgListA8 = imgListA8.to(device)
                imgListB8 = imgListB8.to(device)
                imgListA16 = imgListA16.to(device)
                imgListB16 = imgListB16.to(device) 
                
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                loss = 0
                with torch.set_grad_enabled(phase=='train'):
                    if phase=='train':  model.train()  # backward + optimize only if in training phase                        
                    else: model.eval()
                    
                    PFFx16_2to1, PFFx16_1to2 = Model16(imgListA16, imgListB16)
                    
                    if epoch==0 and iterCount==0:
                        upFeatMapFunc_x2 = nn.Upsample(
                            size=[PFFx16_2to1.size(2)*upFactorX2,
                                  PFFx16_2to1.size(3)*upFactorX2],
                            mode='nearest', align_corners=None)
                        
                        upFeatMapFunc_x4 = nn.Upsample(
                            size=[PFFx16_2to1.size(2)*upFactorX4, 
                                  PFFx16_2to1.size(3)*upFactorX4],
                            mode='nearest', align_corners=None)
                    
                    PFFx16_2to1_up2 = upFeatMapFunc_x2(PFFx16_2to1)
                    PFFx16_1to2_up2 = upFeatMapFunc_x2(PFFx16_1to2)
                    PFFx16_2to1_up2 = upsamplePFF(PFFx16_2to1_up2, ksize, upF_x2, upFactorX2)
                    PFFx16_1to2_up2 = upsamplePFF(PFFx16_1to2_up2, ksize, upF_x2, upFactorX2)

                    
                    recImgBA8 = warpImgWithScale2(imgListB8, PFFx16_2to1)
                    recImgBA8 = recImgBA8.detach()
                    PFFx8_2to1, _ = Model8(imgListA8, recImgBA8)                     
                    recImgA16x4 = warpImgWithScale4(imgListB4, PFFx16_2to1)                                        
                    recImgA8x2 = warpImgWithScale2(recImgA16x4, PFFx8_2to1)                 
                    recImgA8x2 = recImgA8x2.detach()  
                    
                    
                    embFeature2_to_1, embFeature1_to_2 = model(imgListA4, recImgA8x2) 
                    lossRec1to2 = loss_pixelReconstruction(imgListA4, recImgA8x2, embFeature1_to_2)
                    reconsturctedImage2 = loss_pixelReconstruction.reconstructImage
                    lossRec2to1 = loss_pixelReconstruction(recImgA8x2, imgListA4, embFeature2_to_1)
                    reconsturctedImage1 = loss_pixelReconstruction.reconstructImage
                    loss += lossRec1to2
                    loss += lossRec2to1
                    running_loss_reconstruction += lossRec1to2.item()*imgListA4.size(0)
                    running_loss_reconstruction += lossRec2to1.item()*imgListA4.size(0)
                    
                        
                    lossFlow4Recon1to2 = loss_warp4reconstruction(imgListA4, recImgA8x2, embFeature1_to_2)
                    loss += lossFlow4Recon1to2
                    lossFlow4Recon2to1 = loss_warp4reconstruction(recImgA8x2, imgListA4, embFeature2_to_1)
                    loss += lossFlow4Recon2to1
                    running_loss_flow4warpRecon += lossFlow4Recon1to2.item()*imgListB8.size(0)
                    running_loss_flow4warpRecon += lossFlow4Recon2to1.item()*imgListB8.size(0)
                                            
                    
                    lossSmooth2to1 = loss_filterSmoothness(embFeature2_to_1)
                    lossSmooth1to2 = loss_filterSmoothness(embFeature1_to_2)
                    loss += lossSmooth2to1
                    loss += lossSmooth1to2
                    running_loss_filterSmoothness += lossSmooth2to1.item()*imgListA8.size(0)
                    running_loss_filterSmoothness += lossSmooth1to2.item()*imgListA8.size(0)                            
                                       
                    loss_imageGradient2to1 = weight4ImGrad*loss_imageGradient(reconsturctedImage1, 
                                                                              imgListA4)*imgListA8.size(0)
                    loss_imageGradient1to2 = weight4ImGrad*loss_imageGradient(reconsturctedImage2,
                                                                              recImgA8x2)*imgListA8.size(0)
                    loss += loss_imageGradient2to1
                    loss += loss_imageGradient1to2
                    running_loss_imageGradient += loss_imageGradient2to1.item()*imgListA8.size(0)
                    running_loss_imageGradient += loss_imageGradient1to2.item()*imgListA8.size(0)
                        
                        
                        
                        
                        
                        
                        
                        
                        
                    recImgAB8 = warpImgWithScale2(imgListA8, PFFx16_1to2)
                    recImgAB8 = recImgBA8.detach()
                    PFFx8_2to1, _ = Model8(imgListB8, recImgAB8)                     
                    recImgB16x4 = warpImgWithScale4(imgListA4, PFFx16_2to1)                                        
                    recImgB8x2 = warpImgWithScale2(recImgB16x4, PFFx8_2to1)                 
                    recImgB8x2 = recImgA8x2.detach()                      
                    
                    
                    embFeature2_to_1, embFeature1_to_2 = model(imgListB4, recImgB8x2) 
                    lossRec1to2 = loss_pixelReconstruction(imgListB4, recImgB8x2, embFeature1_to_2)
                    reconsturctedImage2 = loss_pixelReconstruction.reconstructImage
                    lossRec2to1 = loss_pixelReconstruction(recImgB8x2, imgListB4, embFeature2_to_1)
                    reconsturctedImage1 = loss_pixelReconstruction.reconstructImage
                    loss += lossRec1to2
                    loss += lossRec2to1
                    running_loss_reconstruction += lossRec1to2.item()*imgListB4.size(0)
                    running_loss_reconstruction += lossRec2to1.item()*imgListB4.size(0)                        
                    
                        
                    lossFlow4Recon1to2 = loss_warp4reconstruction(imgListB4, recImgB8x2, embFeature1_to_2)
                    loss += lossFlow4Recon1to2
                    lossFlow4Recon2to1 = loss_warp4reconstruction(recImgB8x2, imgListB4, embFeature2_to_1)
                    loss += lossFlow4Recon2to1
                    running_loss_flow4warpRecon += lossFlow4Recon1to2.item()*imgListB4.size(0)
                    running_loss_flow4warpRecon += lossFlow4Recon2to1.item()*imgListB4.size(0)
                    
                        
                    lossSmooth2to1 = loss_filterSmoothness(embFeature2_to_1)
                    lossSmooth1to2 = loss_filterSmoothness(embFeature1_to_2)
                    loss += lossSmooth2to1
                    loss += lossSmooth1to2
                    running_loss_filterSmoothness += lossSmooth2to1.item()*imgListA8.size(0)
                    running_loss_filterSmoothness += lossSmooth1to2.item()*imgListA8.size(0)
                        
                    loss_imageGradient2to1 = weight4ImGrad*loss_imageGradient(reconsturctedImage1, 
                                                                              imgListB4)*imgListA8.size(0)
                    loss_imageGradient1to2 = weight4ImGrad*loss_imageGradient(reconsturctedImage2,
                                                                              recImgB8x2)*imgListA8.size(0)
                    loss += loss_imageGradient2to1
                    loss += loss_imageGradient1to2
                    running_loss_imageGradient += loss_imageGradient2to1.item()*imgListA8.size(0)
                    running_loss_imageGradient += loss_imageGradient1to2.item()*imgListA8.size(0)                       
                    
                    
                    if phase=='train': loss.backward()                        
                    if phase=='train': optimizer.step()
                    
                # statistics  
                iterCount += 1
                sampleCount += imgListA8.size(0)                                
                running_loss += loss.item() * imgListA8.size(0)                                
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
                fn.close()
                
                
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    fn = open(log_filename,'a')
    fn.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    fn.close()
   
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
