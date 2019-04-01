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

class PEM(nn.Module):
    def __init__(self, emb_dimension=128, num_class=10, pretrained=True):
        super(PEM, self).__init__()
        self.num_class = num_class
        self.emb_dimension = emb_dimension        
        
        resnet = models.resnet50(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        
                
        # tweak resnet backbone to output features 8x smaller than input image
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
                
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        
        
        self.streamTwo_feats = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
        )
        
        
        
        self.block5_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer4[-1].conv3.out_channels, emb_dimension, 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )            
        self.block5_feats = nn.Sequential(
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension),
            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )
        
        
        
        self.layer1_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer1[-1].conv3.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer1_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
                        
        
        
        self.layer2_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer2[-1].conv3.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer2_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        
        

        self.layer3_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer3[-1].conv3.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer3_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        
        
        
        self.emb = nn.Sequential(
            nn.Conv2d(int(emb_dimension*2.5+32), emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension),
            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )
        
        self.dropout = nn.Dropout(0.7)        
        self.classifier = nn.Conv2d(emb_dimension, num_class, 1)
        
        
    def forward(self, inputs):
        self.interp_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  
        self.interp_x8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  
        
        input_size = inputs.size()
        
        out = self.layer0(inputs)
        out_stream2 = self.streamTwo_feats(inputs)        
        
        
        out = self.layer1(out)        
        out_layer1 = self.layer1_dimRed(out)        
        out_layer1 = self.interp_x4(out_layer1)
        out_layer1 = self.layer2_feats(out_layer1)
        
        
        out = self.layer2(out)
        
        out_layer2 = self.layer2_dimRed(out)        
        out_layer2 = self.interp_x8(out_layer2)
        out_layer2 = self.layer2_feats(out_layer2)
        
        
        out = self.layer3(out)
        
        out_layer3 = self.layer3_dimRed(out)        
        out_layer3 = self.interp_x8(out_layer3)
        out_layer3 = self.layer3_feats(out_layer3)
        
        
        out = self.layer4(out)
        
        out = self.block5_dimRed(out)
        self.interp = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)        
        out = self.interp(out)        
        out = self.block5_feats(out)
        
        out = torch.cat([out_stream2, out_layer1, out_layer2, out_layer3, out], 1)
        out=self.emb(out)
        
        out = self.dropout(out)        
        out = self.classifier(out)
                
        return out      

    
    


    
class Classifier(nn.Module):
    def __init__(self, inputDim=128, num_class=10):
        super(Classifier, self).__init__()
        self.num_class = num_class
        self.inputDim = inputDim        
        self.dropoutRate = 0.5
        
        if self.dropoutRate==0:
            self.feature_for_cls = nn.Sequential(
                nn.Conv2d(inputDim, num_class, 1)
            )
            
        else:
            self.feature_for_cls = nn.Sequential(
                nn.Dropout(self.dropoutRate),
                nn.Conv2d(inputDim, num_class, 1)
            )
        
    def forward(self, inputs):
        out = self.feature_for_cls(inputs)        
        return out        
    
    
    
    
    
class PixelEmbedModel(nn.Module):
    def __init__(self, emb_dimension=128, pretrained=True):
        super(PixelEmbedModel, self).__init__()
        self.emb_dimension = emb_dimension        
        
        resnet = models.resnet50(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        
                
        # tweak resnet backbone to output features 8x smaller than input image
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
                
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
                
        
        self.streamTwo_feats = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
        )
        
                
        self.block5_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer4[-1].conv3.out_channels, emb_dimension, 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )            
        self.block5_feats = nn.Sequential(
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension),
            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )
        
                
        self.layer1_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer1[-1].conv3.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer1_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
                        
                
        self.layer2_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer2[-1].conv3.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer2_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
                

        self.layer3_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer3[-1].conv3.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer3_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        
        
        self.emb = nn.Sequential(
            nn.Conv2d(int(emb_dimension*2.5+32), emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension),
            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )
        
        
    def forward(self, inputs):
        self.interp_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  
        self.interp_x8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  
        
        input_size = inputs.size()
        
        out = self.layer0(inputs)
        out_stream2 = self.streamTwo_feats(inputs)        
        
        out = self.layer1(out)        
        out_layer1 = self.layer1_dimRed(out)        
        out_layer1 = self.interp_x4(out_layer1)
        out_layer1 = self.layer2_feats(out_layer1)
        
        out = self.layer2(out)
        
        out_layer2 = self.layer2_dimRed(out)        
        out_layer2 = self.interp_x8(out_layer2)
        out_layer2 = self.layer2_feats(out_layer2)
        
        
        out = self.layer3(out)
        
        out_layer3 = self.layer3_dimRed(out)        
        out_layer3 = self.interp_x8(out_layer3)
        out_layer3 = self.layer3_feats(out_layer3)
        
        
        out = self.layer4(out)
        
        out = self.block5_dimRed(out)
        self.interp = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)        
        out = self.interp(out)        
        out = self.block5_feats(out)
            
        out = torch.cat([out_stream2, out_layer1, out_layer2, out_layer3, out], 1)
        out=self.emb(out)
                
        return out      
    
    

    
    
    
    
class PixelEmbedModelResNet18(nn.Module):
    def __init__(self, emb_dimension=128, pretrained=True):
        super(PixelEmbedModelResNet18, self).__init__()
        self.emb_dimension = emb_dimension        
        
        resnet = models.resnet18(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        
                
        # tweak resnet backbone to output features 8x smaller than input image
        for n, m in self.layer3.named_modules():
            if 'conv1' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
                
        for n, m in self.layer4.named_modules():
            if 'conv1' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
                
        
        self.streamTwo_feats = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
        )
        
                
        self.block5_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer4[-1].conv2.out_channels, emb_dimension, 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )            
        self.block5_feats = nn.Sequential(
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension),
            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )
        
                
        self.layer1_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer1[-1].conv2.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer1_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
                        
                
        self.layer2_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer2[-1].conv2.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer2_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
                

        self.layer3_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer3[-1].conv2.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer3_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        
        
        self.emb = nn.Sequential(
            nn.Conv2d(int(emb_dimension*2.5+32), emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension),
            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )
        
        
    def forward(self, inputs):
        self.interp_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  
        self.interp_x8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  
        
        input_size = inputs.size()
        
        out = self.layer0(inputs)
        out_stream2 = self.streamTwo_feats(inputs)        
        
        out = self.layer1(out)        
        out_layer1 = self.layer1_dimRed(out)        
        out_layer1 = self.interp_x4(out_layer1)
        out_layer1 = self.layer2_feats(out_layer1)
        
        out = self.layer2(out)
        
        out_layer2 = self.layer2_dimRed(out)        
        out_layer2 = self.interp_x8(out_layer2)
        out_layer2 = self.layer2_feats(out_layer2)
        
        
        out = self.layer3(out)
        
        out_layer3 = self.layer3_dimRed(out)        
        out_layer3 = self.interp_x8(out_layer3)
        out_layer3 = self.layer3_feats(out_layer3)
        
        
        out = self.layer4(out)
        
        out = self.block5_dimRed(out)
        self.interp = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)        
        out = self.interp(out)        
        out = self.block5_feats(out)
            
        #print(out_stream2.size())
        #print(out_layer1.size())
        #print(out_layer2.size())
        #print(out_layer3.size())
        #print(out.size())
        out = torch.cat([out_stream2, out_layer1, out_layer2, out_layer3, out], 1)
        out=self.emb(out)
                
        return out      
    

    
    
class PixelEmbedModelResNet50(nn.Module):
    def __init__(self, emb_dimension=128, pretrained=True):
        super(PixelEmbedModelResNet50, self).__init__()
        self.emb_dimension = emb_dimension        
        
        resnet = models.resnet50(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        
                
        # tweak resnet backbone to output features 8x smaller than input image
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
                
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
                
        
        self.streamTwo_feats = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
        )
        
                
        self.block5_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer4[-1].conv3.out_channels, emb_dimension, 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )            
        self.block5_feats = nn.Sequential(
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension),
            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )
        
                
        self.layer1_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer1[-1].conv3.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer1_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
                        
                
        self.layer2_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer2[-1].conv3.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer2_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
                

        self.layer3_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer3[-1].conv3.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer3_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        
        
        self.emb = nn.Sequential(
            nn.Conv2d(int(emb_dimension*2.5+32), emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension),
            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )
        
        
    def forward(self, inputs):
        self.interp_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  
        self.interp_x8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  
        
        input_size = inputs.size()
        
        out = self.layer0(inputs)
        out_stream2 = self.streamTwo_feats(inputs)        
        
        out = self.layer1(out)        
        out_layer1 = self.layer1_dimRed(out)        
        out_layer1 = self.interp_x4(out_layer1)
        out_layer1 = self.layer2_feats(out_layer1)
        
        out = self.layer2(out)
        
        out_layer2 = self.layer2_dimRed(out)        
        out_layer2 = self.interp_x8(out_layer2)
        out_layer2 = self.layer2_feats(out_layer2)
        
        
        out = self.layer3(out)
        
        out_layer3 = self.layer3_dimRed(out)        
        out_layer3 = self.interp_x8(out_layer3)
        out_layer3 = self.layer3_feats(out_layer3)
        
        
        out = self.layer4(out)
        
        out = self.block5_dimRed(out)
        self.interp = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)        
        out = self.interp(out)        
        out = self.block5_feats(out)
            
        out = torch.cat([out_stream2, out_layer1, out_layer2, out_layer3, out], 1)
        out=self.emb(out)
                
        return out      
    
    

    
    
class PixelEmbedModelTiny(nn.Module):
    def __init__(self, emb_dimension=16, pretrained=False):
        super(PixelEmbedModelTiny, self).__init__()
        self.emb_dimension = emb_dimension        
        '''
        resnet = models.resnet18(pretrained=False)
        layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        layer1= resnet.layer1

        for child in layer0.children():
            if 'Conv2d' in str(type(child)): 
                child.stride=(1,1)
                child.padding=(0,0)
            print(child)

        i = 0
        for n, m in layer1.named_modules():
            print(i, m)
            i+=1
            if 'conv1' in n:
                m.dilation, m.padding, m.stride = (1, 1), (0, 0), (1, 1)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (1, 1), (0, 0), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        '''
        
        self.fea = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(64,momentum=0.001),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(64,momentum=0.001),
            
            nn.Conv2d(64, 32, kernel_size=3, padding=0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32,momentum=0.001),
            
            nn.Conv2d(32, self.emb_dimension, kernel_size=3, padding=0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(self.emb_dimension,momentum=0.001),
        )
        
        
    def forward(self, inputs):        
        input_size = inputs.size()        
        out = self.fea(inputs)
        return out      
    


    
    
    
    
    
        
    
    
class PixelEmbedModelResNet18Shallow(nn.Module):
    def __init__(self, emb_dimension=128, pretrained=True):
        super(PixelEmbedModelResNet18Shallow, self).__init__()
        self.emb_dimension = emb_dimension        
        
        resnet = models.resnet18(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        
                
        # tweak resnet backbone to output features 8x smaller than input image
        for n, m in self.layer3.named_modules():
            if 'conv1' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
                
        for n, m in self.layer4.named_modules():
            if 'conv1' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
                
        
        self.streamTwo_feats = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, padding=4, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
        )
        
                
        self.block5_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer4[-1].conv2.out_channels, emb_dimension, 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )            
        self.block5_feats = nn.Sequential(
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension),
            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )
        
                
        self.layer1_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer1[-1].conv2.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer1_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
                        
                
        self.layer2_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer2[-1].conv2.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer2_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
                

        self.layer3_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer3[-1].conv2.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        self.layer3_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2)),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(int(emb_dimension/2))
        )
        
        
        self.emb = nn.Sequential(
            nn.Conv2d(int(emb_dimension*2.5+32), emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension),
            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(emb_dimension)
        )
        
        
    def forward(self, inputs):
        self.interp_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  
        self.interp_x8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  
        
        input_size = inputs.size()
        
        out = self.layer0(inputs)
        out_stream2 = self.streamTwo_feats(inputs)        
        
        out = self.layer1(out)        
        out_layer1 = self.layer1_dimRed(out)        
        out_layer1 = self.interp_x4(out_layer1)
        out_layer1 = self.layer2_feats(out_layer1)
        
        out = self.layer2(out)
        
        out_layer2 = self.layer2_dimRed(out)        
        out_layer2 = self.interp_x8(out_layer2)
        out_layer2 = self.layer2_feats(out_layer2)
        
        
        out = self.layer3(out)
        
        out_layer3 = self.layer3_dimRed(out)        
        out_layer3 = self.interp_x8(out_layer3)
        out_layer3 = self.layer3_feats(out_layer3)
        
        
        out = self.layer4(out)
        
        out = self.block5_dimRed(out)
        self.interp = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)        
        out = self.interp(out)        
        out = self.block5_feats(out)
            
        #print(out_stream2.size())
        #print(out_layer1.size())
        #print(out_layer2.size())
        #print(out_layer3.size())
        #print(out.size())
        out = torch.cat([out_stream2, out_layer1, out_layer2, out_layer3, out], 1)
        out=self.emb(out)
                
        return out      

        