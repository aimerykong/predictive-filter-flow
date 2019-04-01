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
from torchvision.models import *
    
    
    
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
    
    
    
class ResNet4MultigridPFF(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(ResNet4MultigridPFF, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 196, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(196 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def compactResNet18forMultigridPFF(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet4MultigridPFF(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model




    
    
    
class MultigridPFF_tiny(nn.Module):
    def __init__(self, emb_dimension=128):
        super(MultigridPFF_tiny, self).__init__()
        self.emb_dimension = emb_dimension        
        
        resnet = compactResNet18forMultigridPFF()
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3 = resnet.layer1, resnet.layer2, resnet.layer3
        
                
        # tweak resnet backbone to output features 8x smaller than input image
        for n, m in self.layer3.named_modules():
            if 'conv1' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        self.streamTwo_feats = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, momentum=0.001),
            nn.ReLU(True),
            
            nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(16, momentum=0.001),
            nn.ReLU(True),
            
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, momentum=0.001),
            nn.ReLU(True),
            
            nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(16, momentum=0.001),
            nn.ReLU(True),
            
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16, momentum=0.001),
            nn.ReLU(True),
        )
        
                
        
        
                
        self.layer1_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer1[-1].conv2.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(int(emb_dimension/2), momentum=0.001),
            nn.ReLU(True),
        )
        self.layer1_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(emb_dimension/2), momentum=0.001),
            nn.ReLU(True),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(emb_dimension/2), momentum=0.001),
            nn.ReLU(True),
        )
                        
                
        self.layer2_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer2[-1].conv2.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(int(emb_dimension/2), momentum=0.001),
            nn.ReLU(True),
        )
        self.layer2_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(emb_dimension/2), momentum=0.001),
            nn.ReLU(True),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(emb_dimension/2), momentum=0.001),
            nn.ReLU(True),
        )
                

        self.layer3_dimRed = nn.Sequential(
            nn.Conv2d(resnet.layer3[-1].conv2.out_channels, int(emb_dimension/2), 
                      kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(int(emb_dimension/2), momentum=0.001),
            nn.ReLU(True),
        )
        self.layer3_feats = nn.Sequential(
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(emb_dimension/2), momentum=0.001),
            nn.ReLU(True),
            
            nn.Conv2d(int(emb_dimension/2), int(emb_dimension/2), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(emb_dimension/2), momentum=0.001),
            nn.ReLU(True),            
        )
                
        self.emb = nn.Sequential(
            nn.Conv2d(int(emb_dimension*1.5+16), emb_dimension, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(emb_dimension, momentum=0.001),
            nn.ReLU(True),
            
            nn.Conv2d(emb_dimension, emb_dimension, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(emb_dimension, momentum=0.001),
            nn.ReLU(True),
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

        out = torch.cat([out_stream2, out_layer1, out_layer2, out_layer3], 1)
        out = self.emb(out)
                
        return out