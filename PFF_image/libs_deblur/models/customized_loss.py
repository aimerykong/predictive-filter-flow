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
#from models.pixel_embedding_model import *
#from models.segm_basic_model import *
#from models.customized_loss import *

class cosSimLoss(nn.Module):
    def __init__(self, randNum=-100, device='cpu', margin=0.4,
                 size_average=None, reduction='elementwise_mean'):
        super(cosSimLoss, self).__init__()
        self.weight = None
        self.size_average = size_average
        self.reduction = reduction        
        self.randNum = randNum
        self.device = device
        self.randIdx_H = 0
        self.randIdx_W = 0
        self.margin = margin
        
    def forward(self, inputs, target): 
        weightsFromNum = target.clone()
        weightsFromNum[weightsFromNum!=0] = 1
        weightsFromNum[weightsFromNum==0] = 0.05
        
        if self.randNum>0:
            inputs_size = inputs.size()
            tensor_way = len(inputs_size)
            self.randIdx_W = np.random.choice(inputs_size[-1], self.randNum, replace=False)
            self.randIdx_W = torch.from_numpy(self.randIdx_W).long().to(self.device)
            self.randIdx_H = np.random.choice(inputs_size[-2], self.randNum, replace=False)
            self.randIdx_H = torch.from_numpy(self.randIdx_H).long().to(self.device)

            inputs = self.rand_sample_pixels(inputs)
            target = self.rand_sample_pixels(target)
            weightsFromNum = self.rand_sample_pixels(weightsFromNum)
            
            
        inputs_NCM = inputs.view(inputs.size(0),inputs.size(1),-1) # NxCxHxW --> NxCxM, where M=H*W
        inputs_NMC = inputs_NCM.permute(0,2,1) # NxCxM --> NxMxC, permute axes        
        cosSimMat = torch.matmul(inputs_NMC, inputs_NCM)
        cosSimMat = cosSimMat.clamp(-1,1)# torch.clamp(input, min, max, out=None)
        cosSimMat.add_(1).mul_(0.5)
        
        
        # findicator vector --> pair-wise binary matrix showing whether two points have the same label
        target_simMat = self.indicator_to_similarity_matrix(target)

        # generate per-pixel weight 
        weightsFromNum = self.gen_per_pixel_weight_matrix(weightsFromNum)
        
        # computing the loss over cosSimMat and target_simMat (0: inter-obj, 1:inner-obj)
        cosSimMat = cosSimMat.view(cosSimMat.size(0),-1)
        target_simMat = target_simMat.view(target_simMat.size(0),-1)
        weightsFromNum = weightsFromNum.view(weightsFromNum.size(0),-1)
        
        totalNum = target_simMat.size(0)*target_simMat.size(1)+2
        posNum = torch.sum(target_simMat.view(-1))+1
        negNum = totalNum-posNum +1
        weight_neg = 1.0/negNum
        weight_pos = 1.0/posNum
          
        
        posPairLoss = torch.mul(1-cosSimMat, target_simMat) * weightsFromNum
        
        #negPairLoss = torch.mul(cosSimMat, 1-target_simMat)        
        negPairLoss = (cosSimMat-self.margin) * (1-target_simMat) * weightsFromNum    
        negPairLoss = negPairLoss.clamp(min=0)
                
        posPairLoss = torch.sum(torch.sum(torch.sum(posPairLoss))) * weight_pos
        negPairLoss = torch.sum(torch.sum(torch.sum(negPairLoss))) * weight_neg
        
        lossValue = posPairLoss + negPairLoss
        return lossValue        
    
    
    def gen_per_pixel_weight_matrix(self, weight):
        # generate the per-pixel weight for training
        weight_NxM = weight.view(weight.size(0), -1) # NxHxW --> NxM
        weight_NxMx1 = weight_NxM.unsqueeze(-1) # NxM --> NxMx1
        weight_NxMx1 = weight_NxMx1.expand(weight_NxM.size(0), weight_NxM.size(1), weight_NxM.size(1))        
        #weight_Nx1xM = weight_NxM.unsqueeze(1) # NxM --> Nx1xM
        #weight_Nx1xM = weight_Nx1xM.expand(weight_Nx1xM.size(0), weight_Nx1xM.size(2), weight_Nx1xM.size(2))
        #weight_simMat = torch.mul(weight_NxMx1, weight_Nx1xM)
        weight_simMat = torch.mul(weight_NxMx1, weight_NxMx1.permute(0,2,1))        
        weight_simMat = weight_simMat.type('torch.FloatTensor')
        return weight_simMat.to(self.device)
    
    
    def indicator_to_similarity_matrix(self, target):
        # per-pixel weight 
        WeightMat = target.clone()
        WeightMat[WeightMat!=0] = 5
        WeightMat[WeightMat==0] = 1
        
        # generate the target for training
        target_NxM = target.view(target.size(0), -1) # NxHxW --> NxM
        target_NxMx1 = target_NxM.unsqueeze(-1) # NxM --> NxMx1
        target_NxMx1 = target_NxMx1.expand(target_NxM.size(0), target_NxM.size(1), target_NxM.size(1))        
        target_Nx1xM = target_NxM.unsqueeze(1) # NxM --> Nx1xM
        target_Nx1xM = target_Nx1xM.expand(target_Nx1xM.size(0), target_Nx1xM.size(2), target_Nx1xM.size(2))        
        target_simMat = target_NxMx1.eq(target_Nx1xM)
        target_simMat = target_simMat.type('torch.FloatTensor')
        return target_simMat.to(self.device)
    
    
            
    def rand_sample_pixels(self, inputs):
        inputs_size = inputs.size()
        tensor_way = len(inputs_size)        
        inputs = torch.index_select(inputs, tensor_way-2, self.randIdx_H, out=None) 
        inputs = torch.index_select(inputs, tensor_way-1, self.randIdx_W, out=None) 
        return inputs