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


class PixelEmbedModel4SemanticSegm(nn.Module):
    def __init__(self, emb_dimension=128, num_class=10, pretrained=True):
        super(PixelEmbedModel4SemanticSegm, self).__init__()
        self.num_class = num_class
        self.emb_dimension = emb_dimension
        self.embFeature = 0
        self.L2normFeature = 0
        
        self.PEMbase = PixelEmbedModel(emb_dimension=self.emb_dimension, pretrained=False)   
        self.dropout = nn.Dropout(0.7)        
        self.classifier = nn.Conv2d(self.emb_dimension, self.num_class, 1)
        
        
    def forward(self, inputs):
        input_size = inputs.size()
        
        self.embFeature = self.PEMbase.forward(inputs)
        self.L2normFeature = F.normalize(self.embFeature, p=2, dim=1) # normalize
        out = self.dropout(self.L2normFeature)
        out = self.classifier(out)
        
        return out
    
class PixelEmbedModel4InstSeg(nn.Module):
    def __init__(self, emb_dimension=128, pretrained=False):
        super(PixelEmbedModel4InstSeg, self).__init__()
        self.emb_dimension = emb_dimension  
        self.PEMbase = PixelEmbedModel(emb_dimension=self.emb_dimension, pretrained=pretrained)  
        self.rawEmbFeature = 0
        self.embFeature = 0
        self.L2normFeature = 0
        
        self.emb_before_L2normalization = nn.Sequential(            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=1, padding=0, bias=True)
        )
        
    def forward(self, inputs):        
        self.rawEmbFeature = self.PEMbase.forward(inputs)
        self.embFeature = self.emb_before_L2normalization(self.rawEmbFeature)
        self.L2normFeature = F.normalize(self.embFeature, p=2, dim=1) # normalize
        return self.L2normFeature                        

    
    
class PixelEmbedModel4InstSegWithCartesian(nn.Module):
    def __init__(self, emb_dimension=128, device='cpu', pretrained=False):
        super(PixelEmbedModel4InstSegWithCartesian, self).__init__()
        self.emb_dimension = emb_dimension  
        self.PEMbase = PixelEmbedModel(emb_dimension=self.emb_dimension, pretrained=pretrained)  
        self.rawEmbFeature = 0
        self.embFeature = 0
        self.L2normFeature = 0
        self.device=device
        
        self.emb_transform = nn.Sequential(            
            nn.Conv2d(emb_dimension+4, emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension),
            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=1, padding=0, bias=True)
        )
        
    def forward(self, inputs):        
        self.rawEmbFeature = self.PEMbase.forward(inputs)        
        
        inputs_size = self.rawEmbFeature.size()
        H, W = inputs_size[2], inputs_size[3]
        yv, xv = torch.meshgrid([torch.arange(0,H), torch.arange(0,W)])
        yv = yv.type('torch.FloatTensor') / H
        xv = xv.type('torch.FloatTensor') / W
        X = xv.numpy()
        xv_fliplr = X[:,::-1].copy()
        xv_fliplr = torch.from_numpy(xv_fliplr)
        Y = yv.numpy()
        yv_flipud = Y[::-1].copy()
        yv_flipud = torch.from_numpy(yv_flipud)
        
        
        xv_fliplr = xv_fliplr.expand(inputs_size[0],1,H,W).to(self.device)
        xv = xv.expand(inputs_size[0],1,H,W).to(self.device) 
        yv = yv.expand(inputs_size[0],1,H,W).to(self.device)
        yv_flipud = yv_flipud.expand(inputs_size[0],1,H,W).to(self.device) 
        
                
        out = torch.cat([self.rawEmbFeature, xv, xv_fliplr, yv, yv_flipud], 1)
        
        self.embFeature = self.emb_transform(out)
        self.L2normFeature = F.normalize(self.embFeature, p=2, dim=1) # normalize
        return self.L2normFeature       
    