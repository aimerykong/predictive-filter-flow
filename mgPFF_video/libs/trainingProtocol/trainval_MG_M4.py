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





def train_model(model, Model16, dataloaders, dataset_sizes, 
                loss_pixelReconstruction,
                loss_pixelReconstructionX2,
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

    
    upFactorX = 2
    FFsize = ksize = 11    
    upF = torch.zeros(FFsize*upFactorX, FFsize)
    for i in range(upFactorX):
        idxRows = list(range(i,upF.size(0), upFactorX))
        upF[idxRows, :] = torch.eye(FFsize)
    upF = upF.to(device)

        
        
        
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
            running_loss_filterSmoothness = 0.0
            running_loss_groupSparsity = 0.0
            running_loss_imageGradient = 0.0
            running_loss_imageGradientX2 = 0.0
            
            
            # Iterate over data.
            iterCount,sampleCount = 0, 0
            for sample in dataloaders[phase]:
                imgListA8, imgListB8, imgListA16, imgListB16 = sample
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
                    
                    PFF2to1, PFF1to2 = Model16(imgListA16, imgListB16)
                    if epoch==0 and iterCount==0:
                        upFeatMapFunc = nn.Upsample(size=[PFF2to1.size(2)*upFactorX, 
                                                          PFF2to1.size(3)*upFactorX],
                                                    mode='nearest', align_corners=None)
                    
                    PFF2to1_up2 = upFeatMapFunc(PFF2to1)                    
                    PFF1to2_up2 = upFeatMapFunc(PFF1to2)                    
                    PFF2to1_up2 = upsamplePFF(PFF2to1_up2, ksize, upF, upFactorX)
                    PFF1to2_up2 = upsamplePFF(PFF1to2_up2, ksize, upF, upFactorX)

                    _ = loss_pixelReconstructionX2(imgListA8, imgListB8, PFF1to2_up2)
                    recImgAB8 = loss_pixelReconstructionX2.reconstructImage                        
                    _ = loss_pixelReconstructionX2(imgListB8, imgListA8, PFF2to1_up2)
                    recImgBA8 = loss_pixelReconstructionX2.reconstructImage
                    
                    recImgAB8 = recImgAB8.detach()
                    recImgBA8 = recImgBA8.detach()
                    
                                       
                    
                    embFeature2_to_1, embFeature1_to_2 = model(recImgAB8, imgListB8) 
                    lossRec1to2 = loss_pixelReconstruction(recImgAB8, imgListB8, embFeature1_to_2)
                    reconsturctedImage2 = loss_pixelReconstruction.reconstructImage                    
                    lossRec2to1 = loss_pixelReconstruction(imgListB8, recImgAB8, embFeature2_to_1)
                    reconsturctedImage1 = loss_pixelReconstruction.reconstructImage
                    loss += lossRec1to2
                    loss += lossRec2to1
                    running_loss_reconstruction += lossRec1to2.item()*imgListB8.size(0)
                    running_loss_reconstruction += lossRec2to1.item()*imgListB8.size(0)
                                        
                    
                    lossSmooth2to1 = loss_filterSmoothness(embFeature2_to_1)
                    lossSmooth1to2 = loss_filterSmoothness(embFeature1_to_2)
                    loss += lossSmooth2to1
                    loss += lossSmooth1to2
                    running_loss_filterSmoothness += lossSmooth2to1.item()*imgListA8.size(0)
                    running_loss_filterSmoothness += lossSmooth1to2.item()*imgListA8.size(0)                            
                                       
                    loss_imageGradient2to1 = weight4ImGrad*loss_imageGradient(reconsturctedImage1, 
                                                                              recImgAB8)*imgListA8.size(0)
                    loss_imageGradient1to2 = weight4ImGrad*loss_imageGradient(reconsturctedImage2,
                                                                              imgListB8)*imgListA8.size(0)
                    loss += loss_imageGradient2to1
                    loss += loss_imageGradient1to2
                    running_loss_imageGradient += loss_imageGradient2to1.item()*imgListA8.size(0)
                    running_loss_imageGradient += loss_imageGradient1to2.item()*imgListA8.size(0)
                        
                        
                        
                        
                        
                    
                    embFeature2_to_1, embFeature1_to_2 = model(imgListA8, recImgBA8) 
                    lossRec1to2 = loss_pixelReconstruction(imgListA8, recImgBA8, embFeature1_to_2)
                    reconsturctedImage2 = loss_pixelReconstruction.reconstructImage                    
                    lossRec2to1 = loss_pixelReconstruction(recImgBA8, imgListA8, embFeature2_to_1)
                    reconsturctedImage1 = loss_pixelReconstruction.reconstructImage
                    loss += lossRec1to2
                    loss += lossRec2to1
                    running_loss_reconstruction += lossRec1to2.item()*imgListA8.size(0)
                    running_loss_reconstruction += lossRec2to1.item()*imgListA8.size(0)
                                            
                        
                    lossSmooth2to1 = loss_filterSmoothness(embFeature2_to_1)
                    lossSmooth1to2 = loss_filterSmoothness(embFeature1_to_2)
                    loss += lossSmooth2to1
                    loss += lossSmooth1to2
                    running_loss_filterSmoothness += lossSmooth2to1.item()*imgListA8.size(0)
                    running_loss_filterSmoothness += lossSmooth1to2.item()*imgListA8.size(0)                                            
                        
                    loss_imageGradient2to1 = weight4ImGrad*loss_imageGradient(reconsturctedImage1, 
                                                                              imgListA8)*imgListA8.size(0)
                    loss_imageGradient1to2 = weight4ImGrad*loss_imageGradient(reconsturctedImage2,
                                                                              recImgBA8)*imgListA8.size(0)
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
                
                       
                del loss
                if iterCount%100==0:
                    print('\t{}/{} loss: {:.4f} Rec:{:.3f}, Smooth:{:.3f}, imGrad:{:.3f}'.
                          format(
                              iterCount, 
                              len(dataloaders[phase]), 
                              print2screen_avgLoss, print2screen_avgLoss_Rec,
                              print2screen_avgLoss_Smooth,
                              print2screen_avgLoss_imgGrad)                          
                         )
                    fn = open(log_filename,'a')        
                    fn.write('\t{}/{} loss: {:.4f} Rec:{:.3f}, Smooth:{:.3f}, imGrad:{:.3f}\n'.
                             format(
                                 iterCount, 
                                 len(dataloaders[phase]), 
                                 print2screen_avgLoss, print2screen_avgLoss_Rec,
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




def eval_model(model, dataloaders, dataset_sizes, criterion, device='cpu'):
    since = time.time()
    phase = 'val'
    model.eval()   # Set model to evaluate mode
    
    running_loss = 0.0
    # Iterate over data.
    iterCount,sampleCount = 0, 0
    for sample in dataloaders[phase]:
        path_to_sample, img1, img2 = sample
        img1.to(device)
        img2.to(device)
                
        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase=='train'):
            embFeature2_to_1, embFeature1_to_2 = model(img1, img2)  
            
            loss = loss_1_to_2(img1, img2, embFeature1_to_2)            
            loss += loss_2_to_1(img2, img1, embFeature2_to_1)
            
            # statistics  
            iterCount += 1
            sampleCount += img1.size(0)
                                
            running_loss += loss.item() * img1.size(0)
            #running_corrects += torch.sum(preds==labels.data).double()/preds.size(1)/preds.size(2)
            #running_iou += miou(preds, labels.data, n_classes=outputs.size(1)) * preds.size(0)
            
            summary_loss = running_loss / dataset_sizes[phase]
            #summary_acc = running_corrects.double() / dataset_sizes[phase]
            #summary_iou = running_iou / dataset_sizes[phase]
            #summary_iou = summary_iou.mean()
    
    time_elapsed = time.time() - since
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    #print('loss: {:4f}, acc: {:4f}'.format(summary_loss,summary_acc))
    print('loss: {:6f}'.format(summary_loss))
    
    return 


'''
if True:
                        embFeature2_to_1, embFeature1_to_2 = model(imgListA, imgListB)    
                        embFeature2_to_1_up2, embFeature1_to_2_up2 = model.embFeature2_to_1_up2, model.embFeature1_to_2_up2   
                        
                        if epoch==0 and iterCount==0:
                            upFeatMapFunc = nn.Upsample(size=[embFeature2_to_1.size(2)*upFactorX, 
                                                              embFeature2_to_1.size(3)*upFactorX],
                                                        mode='nearest', align_corners=None)
                        
                        
                        pre2to1_up2 = embFeature2_to_1 # filter flow array
                        pre2to1_up2 = upFeatMapFunc(pre2to1_up2)
                        pre1to2_up2 = embFeature1_to_2 # filter flow array
                        pre1to2_up2 = upFeatMapFunc(pre1to2_up2)
        
                        
                        
                        NCHW = pre2to1_up2.size()                        
                        pre2to1_up2 = pre2to1_up2.view(NCHW[0], ksize, ksize, NCHW[2]*NCHW[3])
                        pre2to1_up2 = torch.matmul(upF, pre2to1_up2)
                        pre2to1_up2 = pre2to1_up2.permute(0,2,1,3)
                        pre2to1_up2 = torch.matmul(upF, pre2to1_up2)
                        pre2to1_up2 = pre2to1_up2.permute(0,2,1,3)
                        pre2to1_up2 = pre2to1_up2.view(NCHW[0], ksize*upFactorX, ksize*upFactorX, NCHW[2], NCHW[3])
                        pre2to1_up2 = pre2to1_up2.contiguous().view(NCHW[0],
                                                                    ksize*upFactorX*ksize*upFactorX, NCHW[2], NCHW[3])
                        pre2to1_up2 = pre2to1_up2/upFactorX/upFactorX
                        
                        
                        NCHW = pre1to2_up2.size()                        
                        pre1to2_up2 = pre1to2_up2.view(NCHW[0], ksize, ksize, NCHW[2]*NCHW[3])
                        pre1to2_up2 = torch.matmul(upF, pre1to2_up2)
                        pre1to2_up2 = pre1to2_up2.permute(0,2,1,3)
                        pre1to2_up2 = torch.matmul(upF, pre1to2_up2)
                        pre1to2_up2 = pre1to2_up2.permute(0,2,1,3)
                        pre1to2_up2 = pre1to2_up2.view(NCHW[0], ksize*upFactorX, ksize*upFactorX, NCHW[2], NCHW[3])
                        pre1to2_up2 = pre1to2_up2.contiguous().view(NCHW[0], 
                                                                    ksize*upFactorX*ksize*upFactorX, NCHW[2], NCHW[3])
                        pre1to2_up2 = pre1to2_up2/upFactorX/upFactorX
                        
                        
                        
                        
                        lossRec1to2 = loss_pixelReconstruction(imgListA8, imgListB8, embFeature1_to_2)
                        reconsturctedImage2 = loss_pixelReconstruction.reconstructImage                        
                        lossRec2to1 = loss_pixelReconstruction(imgListB8, imgListA8, embFeature2_to_1)
                        reconsturctedImage1 = loss_pixelReconstruction.reconstructImage
                        loss += lossRec1to2
                        loss += lossRec2to1
                        running_loss_reconstruction += lossRec1to2.item()*imgListA8.size(0)
                        running_loss_reconstruction += lossRec2to1.item()*imgListA8.size(0)
                        
                        
                        
                        _ = loss_pixelReconstructionX2(imgListA4, imgListB4, pre1to2_up2, phase)
                        recImgB_x2 = loss_pixelReconstructionX2.reconstructImage                        
                        _ = loss_pixelReconstructionX2(imgListB4, imgListA4, pre2to1_up2, phase)
                        recImgA_x2 = loss_pixelReconstructionX2.reconstructImage
                        
                        
                        
                        lossRec1to2_up2 = loss_pixelReconstruction(recImgB_x2, imgListB4, embFeature1_to_2_up2)
                        reconsturctedImage2_up2 = weight4ImReconX2*loss_pixelReconstruction.reconstructImage                        
                        lossRec2to1_up2 = loss_pixelReconstruction(recImgA_x2, imgListA4, embFeature2_to_1_up2)
                        reconsturctedImage1_up2 = weight4ImReconX2*loss_pixelReconstruction.reconstructImage
                        loss += lossRec1to2_up2
                        loss += lossRec2to1_up2                      
                        running_loss_reconstructionX2 += lossRec1to2_up2.item()*imgListA4.size(0)
                        running_loss_reconstructionX2 += lossRec2to1_up2.item()*imgListA4.size(0)
                        
                        
                        lossSmooth2to1 = loss_filterSmoothness(embFeature2_to_1)
                        lossSmooth1to2 = loss_filterSmoothness(embFeature1_to_2)
                        loss += lossSmooth2to1
                        loss += lossSmooth1to2
                        running_loss_filterSmoothness += lossSmooth2to1.item()*imgListA8.size(0)
                        running_loss_filterSmoothness += lossSmooth1to2.item()*imgListA8.size(0)
                        
                        
                        loss_groupSparse2to1 = loss_groupSparsity(embFeature2_to_1)*1
                        loss_groupSparse1to2 = loss_groupSparsity(embFeature1_to_2)*1
                        loss += loss_groupSparse2to1
                        loss += loss_groupSparse1to2
                        running_loss_groupSparsity += loss_groupSparse2to1.item()*imgListA8.size(0)
                        running_loss_groupSparsity += loss_groupSparse1to2.item()*imgListA8.size(0)
                        
                        
                        loss_imageGradient2to1 = loss_imageGradient(reconsturctedImage1, 
                                                                                    imgListA8)*imgListA8.size(0)
                        loss_imageGradient1to2 = loss_imageGradient(reconsturctedImage2,
                                                                                    imgListB8)*imgListA8.size(0)
                        loss += loss_imageGradient2to1
                        loss += loss_imageGradient1to2
                        running_loss_imageGradient += loss_imageGradient2to1.item()*imgListA8.size(0)
                        running_loss_imageGradient += loss_imageGradient1to2.item()*imgListA8.size(0)
                        
                        loss_imageGradient2to1X2 = weight4ImGradX2*loss_imageGradient(reconsturctedImage1_up2,
                                                                                      imgListA4)*imgListA4.size(0)
                        loss_imageGradient1to2X2 = weight4ImGradX2*loss_imageGradient(reconsturctedImage2_up2, 
                                                                                      imgListB4)*imgListA4.size(0)
                        loss += loss_imageGradient2to1X2
                        loss += loss_imageGradient1to2X2
                        running_loss_imageGradientX2 += loss_imageGradient2to1X2.item()*imgListA4.size(0)
                        running_loss_imageGradientX2 += loss_imageGradient1to2X2.item()*imgListA4.size(0)
                    
'''