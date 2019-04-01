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
                loss_warp4reconstruction,
                loss_groupSparsity, 
                loss_filterSmoothness,
                loss_imageGradient,
                optimizer, scheduler, 
                num_epochs=25, work_dir='./', device='cpu'):
    
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
            running_loss_groupSparsity = 0.0
            running_loss_imageGradient = 0.0
            
            
            # Iterate over data.
            iterCount,sampleCount = 0, 0
            for sample in dataloaders[phase]:
                #_, _, img1, img2 = sample   
                imgListA16, imgListB16 = sample
                imgListA16 = imgListA16.to(device)
                imgListB16 = imgListB16.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                loss = 0
                with torch.set_grad_enabled(phase=='train'):
                    if phase=='train':  # backward + optimize only if in training phase
                        model.train()                        
                        embFeature2_to_1, embFeature1_to_2 = model(imgListA16, imgListB16)    

                        
                        lossFlow4Recon1to2 = loss_warp4reconstruction(imgListA16,
                                                                      imgListB16, embFeature1_to_2)
                        loss += lossFlow4Recon1to2
                        lossFlow4Recon2to1 = loss_warp4reconstruction(imgListB16,
                                                                      imgListA16, embFeature2_to_1)
                        loss += lossFlow4Recon2to1
                        
                        
                        
                        lossRec1to2 = loss_pixelReconstruction(imgListA16, 
                                                               imgListB16, embFeature1_to_2, phase)
                        reconsturctedImage2 = loss_pixelReconstruction.reconstructImage
                        lossRec2to1 = loss_pixelReconstruction(imgListB16, 
                                                               imgListA16, embFeature2_to_1, phase)
                        reconsturctedImage1 = loss_pixelReconstruction.reconstructImage
                        loss += lossRec1to2
                        loss += lossRec2to1
                        running_loss_reconstruction += lossRec1to2.item()*imgListA16.size(0)
                        running_loss_reconstruction += lossRec2to1.item()*imgListA16.size(0)
                        running_loss_flow4warpRecon += lossFlow4Recon1to2.item()*imgListA16.size(0)
                        running_loss_flow4warpRecon += lossFlow4Recon2to1.item()*imgListA16.size(0)
                        
                        
                        
                        lossSmooth2to1 = loss_filterSmoothness(embFeature2_to_1)
                        lossSmooth1to2 = loss_filterSmoothness(embFeature1_to_2)
                        loss += lossSmooth2to1
                        loss += lossSmooth1to2
                        running_loss_filterSmoothness += lossSmooth2to1.item()*imgListA16.size(0)
                        running_loss_filterSmoothness += lossSmooth1to2.item()*imgListA16.size(0)
                        
                        
                        loss_groupSparse2to1 = loss_groupSparsity(embFeature2_to_1)*1
                        loss_groupSparse1to2 = loss_groupSparsity(embFeature1_to_2)*1
                        loss += loss_groupSparse2to1
                        loss += loss_groupSparse1to2
                        running_loss_groupSparsity += loss_groupSparse2to1.item()*imgListA16.size(0)
                        running_loss_groupSparsity += loss_groupSparse1to2.item()*imgListA16.size(0)
                        
                        
                        loss_imageGradient2to1 = loss_imageGradient(reconsturctedImage1,
                                                                    imgListA16)*imgListA16.size(0)
                        loss_imageGradient1to2 = loss_imageGradient(reconsturctedImage2,
                                                                    imgListB16)*imgListA16.size(0)
                        loss += loss_imageGradient2to1
                        loss += loss_imageGradient1to2
                        running_loss_imageGradient += loss_imageGradient2to1.item()*imgListA16.size(0)
                        running_loss_imageGradient += loss_imageGradient1to2.item()*imgListA16.size(0)

                        
                        loss.backward()
                        optimizer.step()
                    else: 
                        model.eval()                        
                        embFeature2_to_1, embFeature1_to_2 = model(imgListA16, imgListB16)
                        
                        
                        lossFlow4Recon1to2 = loss_warp4reconstruction(imgListA16,
                                                                      imgListB16, embFeature1_to_2)
                        loss += lossFlow4Recon1to2
                        lossFlow4Recon2to1 = loss_warp4reconstruction(imgListB16,
                                                                      imgListA16, embFeature2_to_1)
                        loss += lossFlow4Recon2to1
                        
                          
                        lossRec1to2 = loss_pixelReconstruction(imgListA16, 
                                                               imgListB16, embFeature1_to_2, phase)
                        reconsturctedImage2 = loss_pixelReconstruction.reconstructImage
                        lossRec2to1 = loss_pixelReconstruction(imgListB16, 
                                                               imgListA16, embFeature2_to_1, phase)
                        reconsturctedImage1 = loss_pixelReconstruction.reconstructImage
                        loss += lossRec1to2
                        loss += lossRec2to1
                        running_loss_reconstruction += lossRec1to2.item()*imgListA16.size(0)
                        running_loss_reconstruction += lossRec2to1.item()*imgListA16.size(0)
                        running_loss_flow4warpRecon += lossFlow4Recon1to2.item()*imgListA16.size(0)
                        running_loss_flow4warpRecon += lossFlow4Recon2to1.item()*imgListA16.size(0)
                        
                        
                        lossSmooth2to1 = loss_filterSmoothness(embFeature2_to_1)
                        lossSmooth1to2 = loss_filterSmoothness(embFeature1_to_2)
                        loss += lossSmooth2to1
                        loss += lossSmooth1to2
                        running_loss_filterSmoothness += lossSmooth2to1.item()*imgListA16.size(0)
                        running_loss_filterSmoothness += lossSmooth1to2.item()*imgListA16.size(0)
                        
                        
                        loss_groupSparse2to1 = loss_groupSparsity(embFeature2_to_1)*1
                        loss_groupSparse1to2 = loss_groupSparsity(embFeature1_to_2)*1
                        loss += loss_groupSparse2to1
                        loss += loss_groupSparse1to2
                        running_loss_groupSparsity += loss_groupSparse2to1.item()*imgListA16.size(0)
                        running_loss_groupSparsity += loss_groupSparse1to2.item()*imgListA16.size(0)
                        
                        
                        loss_imageGradient2to1 = loss_imageGradient(reconsturctedImage1,
                                                                    imgListA16)*imgListA16.size(0)
                        loss_imageGradient1to2 = loss_imageGradient(reconsturctedImage2,
                                                                    imgListB16)*imgListA16.size(0)
                        loss += loss_imageGradient2to1
                        loss += loss_imageGradient1to2
                        running_loss_imageGradient += loss_imageGradient2to1.item()*imgListA16.size(0)
                        running_loss_imageGradient += loss_imageGradient1to2.item()*imgListA16.size(0)
                        

                # statistics  
                iterCount += 1
                sampleCount += imgListA16.size(0)                                
                running_loss += loss.item() * imgListA16.size(0)                                
                print2screen_avgLoss = running_loss/sampleCount
                print2screen_avgLoss_Rec = running_loss_reconstruction/sampleCount
                print2screen_avgLoss_flow4warpRecon = running_loss_flow4warpRecon/sampleCount
                print2screen_avgLoss_Smooth = running_loss_filterSmoothness/sampleCount
                print2screen_avgLoss_Sparse = running_loss_groupSparsity/sampleCount
                print2screen_avgLoss_imgGrad = running_loss_imageGradient/sampleCount
                
                
                       
                del loss
                if iterCount%100==0:
                    print('\t{}/{} l:{:.4f}, Rec:{:.3f}, FVrec:{:.3f} smo:{:.3f}, spa:{:.3f}, imgrad:{:.3f}'.
                          format(
                              iterCount, 
                              len(dataloaders[phase]), 
                              print2screen_avgLoss, 
                              print2screen_avgLoss_Rec, 
                              print2screen_avgLoss_flow4warpRecon,
                              print2screen_avgLoss_Smooth,
                              print2screen_avgLoss_Sparse,
                              print2screen_avgLoss_imgGrad)                          
                         )
                    fn = open(log_filename,'a')        
                    fn.write('\t{}/{} l:{:.4f}, Rec:{:.3f}, FVrec:{:.3f} smo:{:.3f}, spa:{:.3f}, imgrad:{:.3f}\n'.
                             format(
                                 iterCount, 
                                 len(dataloaders[phase]), 
                                 print2screen_avgLoss, 
                                 print2screen_avgLoss_Rec, 
                                 print2screen_avgLoss_flow4warpRecon,
                                 print2screen_avgLoss_Smooth,
                                 print2screen_avgLoss_Sparse,
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
