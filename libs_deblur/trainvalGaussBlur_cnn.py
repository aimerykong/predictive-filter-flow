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
from libs_deblur.utils.metrics import *
from libs_deblur.models.pixel_embedding_model import *


def train_model(model, dataloaders, dataset_sizes, loss_1_to_2, optimizer, scheduler, 
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
            # Iterate over data.
            iterCount,sampleCount = 0, 0
            for sample in dataloaders[phase]:
                path_to_sample, img1, img2 = sample                                
                img1 = img1.to(device)
                img2 = img2.to(device)
                    
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train                
                with torch.set_grad_enabled(phase=='train'):
                    if phase=='train':  # backward + optimize only if in training phase
                        model.train()                        
                        reconstImg2 = model(img1)                          
                        loss = loss_1_to_2(reconstImg2, img2)
                        
                        loss.backward()
                        optimizer.step()
                    else: 
                        model.eval()                        
                        reconstImg2 = model(img1)
                        loss = loss_1_to_2(reconstImg2, img2)
                        

                # statistics  
                iterCount += 1
                sampleCount += img1.size(0)                                
                running_loss += loss.item() * img1.size(0)                                
                print2screen_avgLoss = running_loss/sampleCount
                                               
                if iterCount%50==0:
                    print('\t{}/{} loss: {:.6f}'.format(iterCount, len(dataloaders[phase]), print2screen_avgLoss))
                    fn = open(log_filename,'a')        
                    fn.write('\t{}/{} loss: {:.6f}\n'.format(iterCount, len(dataloaders[phase]), print2screen_avgLoss))
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
