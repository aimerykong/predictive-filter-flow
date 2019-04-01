import torch
import numpy as np
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable


def colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r,g,b)) 
    return ret

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]



class DemoShowFlow():
    def __init__(self, height=151, width=151): # kernel size = [height, width]
        self.colorwheel = 0
        self.makeColorwheel()
        
        truerange = 1
        self.height = height
        self.width = height
        validRange = truerange * 1.04
        
        s2 = np.floor(height/2)
        s1 = np.floor(width/2)
        nx, ny = (width, height)
        #xv, yv = np.asarray(range(nx)), np.asarray(range(ny))
        #xv, yv = np.meshgrid(xv, yv)
        xv, yv = np.asarray(range(nx)), np.asarray(range(ny))
        yv, xv = np.meshgrid(yv, xv)

        u = yv*validRange/s2 - validRange
        v = xv*validRange/s1 - validRange
        u /= truerange
        v /= truerange
        
        
        self.FlowColorChart = self.computeColor(u, v)
        self.FlowColorChart[int(s2),:,:] = 0
        self.FlowColorChart[:,int(s2),:] = 0
        self.FlowColorChart /= 255.        
        self.FlowColorChartNoAxes = self.computeColor(u, v, flagOutErase=False)
        self.FlowColorChartNoAxes /= 255.
        
        
    def filterFlow2UV(self, offsetTensor): # in pytorch tensor format
        kernelSize = offsetTensor.size(0)**0.5
        if kernelSize%2==1:
            kernelSize = int(offsetTensor.size(0)**0.5/2)
            #xv, yv = torch.meshgrid([torch.arange(-kernelSize,kernelSize+1), torch.arange(-kernelSize,kernelSize+1)])
            yv, xv = torch.meshgrid([torch.arange(-kernelSize,kernelSize+1), torch.arange(-kernelSize,kernelSize+1)])
        else:
            kernelSize = int(offsetTensor.size(0)**0.5/2)
            #xv, yv = torch.meshgrid([torch.arange(-kernelSize,kernelSize), torch.arange(-kernelSize,kernelSize)])
            yv, xv = torch.meshgrid([torch.arange(-kernelSize,kernelSize), torch.arange(-kernelSize,kernelSize)])
        
        yv, xv = yv.unsqueeze(0).type('torch.FloatTensor'), xv.unsqueeze(0).type('torch.FloatTensor')
        flowMap = offsetTensor.unsqueeze(0)

        yv = yv.contiguous().view(1,-1)
        yv = yv.unsqueeze(-1).unsqueeze(-1)    
        flowMap1 = torch.mul(flowMap,yv) # y   
        flowMap1 = torch.sum(flowMap1,1)

        xv = xv.contiguous().view(1,-1)
        xv = xv.unsqueeze(-1).unsqueeze(-1)    
        flowMap2 = torch.mul(flowMap,xv) # x
        flowMap2 = torch.sum(flowMap2,1)

        flowMap = torch.cat([flowMap2,flowMap1],0) # [x,y]
        return flowMap



    def computeColor(self, u, v, flagOutErase=True):        
        ncols = self.colorwheel.shape[0]
        height, width = u.shape[0:2]

        nanIdx = np.isnan(u) | np.isnan(v)

        rad = np.sqrt(u**2+v**2)
        a = np.arctan2(-v, -u)/np.pi

        fk = (a+1) /2 * (ncols-1)  # -1~1 maped to 1~ncols   
        k0 = np.floor(fk).astype(int)
        k1 = k0+1

        k1[k1==ncols] = 1
        f = fk - k0

        img = np.zeros((height,width,self.colorwheel.shape[1]))
        for i in range(self.colorwheel.shape[1]):
            tmp = self.colorwheel[:,i]    
            col0 = tmp[k0]/255.
            col1 = tmp[k1]/255.

            col = np.multiply(1-f,col0) + np.multiply(f,col1)
            if flagOutErase:
                idx = rad<=1   
                col[idx] = 1 - np.multiply(rad[idx], 1-col[idx])   # increase saturation with radius
                col[~idx] = col[~idx]*0+1   # 0.75   # out of range
            img[:,:, i] = np.floor(255*np.multiply(col,(1-nanIdx)))

        return img

    

    def makeColorwheel(self):
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR
        colorwheel = np.zeros((ncols, 3)) # r g b

        col = 0
        #RY
        colorwheel[0:RY, 0] = 255;
        colorwheel[0:RY, 1] = np.floor(255*np.asarray(range(0,RY))/RY)
        col = col+RY
        #YG
        colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.asarray(range(0,YG))/YG) 
        colorwheel[col:col+YG, 1] = 255
        col = col+YG
        #GC
        colorwheel[col:col+GC, 1] = 255
        colorwheel[col:col+GC, 2] = np.floor(255*np.asarray(range(0,GC))/GC) 
        col = col+GC
        #CB
        colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.asarray(range(0,CB))/CB) 
        colorwheel[col:col+CB, 2] = 255
        col = col+CB
        #BM
        colorwheel[col:col+BM, 2] = 255
        colorwheel[col:col+BM, 0] = np.floor(255*np.asarray(range(0,BM))/BM) 
        col = col+BM
        #MR
        colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.asarray(range(0,MR))/MR) 
        colorwheel[col:col+MR, 0] = 255
        self.colorwheel = colorwheel




def LKOF_RGB(I1g, I2g, window_size, tau=1e-2):
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])#*.25
    w = int(np.floor(window_size/2)) # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    
    mmin = min(I1g.min(),I2g.min())*1.0
    I1g, I2g = I1g-mmin, I2g-mmin # normalize pixels
    cc = max(I1g.max(),I2g.max())*1.0
    I1g, I2g = I1g/cc, I2g/cc # normalize pixels
    
    channelSize = I1g.shape[-1]
    H, W = I1g.shape[0], I1g.shape[1]
    if len(I1g.shape)<3: 
        channelSize = 1
        I1g = np.expand_dims(I1g, axis=2)
        I2g = np.expand_dims(I2g, axis=2)
        
    
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fxTensor = np.zeros((H,W,channelSize),dtype=float)
    fyTensor = np.zeros((H,W,channelSize),dtype=float)
    ftTensor = np.zeros((H,W,channelSize),dtype=float)
    for i in range(channelSize):
        fx = signal.convolve2d(I1g[:,:,i], kernel_x, boundary='symm', mode=mode)
        fy = signal.convolve2d(I1g[:,:,i], kernel_y, boundary='symm', mode=mode)
        ft = signal.convolve2d(I2g[:,:,i], kernel_t, 
                               boundary='symm', mode=mode) + signal.convolve2d(I1g[:,:,i], -kernel_t, 
                                                                               boundary='symm', mode=mode)
        fxTensor[:,:,i] = fx
        fyTensor[:,:,i] = fy
        ftTensor[:,:,i] = ft
    
    u = np.zeros((H,W))
    v = np.zeros((H,W))
    # within window window_size * window_size
    
    for i in range(w, I1g.shape[0]-w):
        for j in range(w, I1g.shape[1]-w):
            Ix = fxTensor[i-w:i+w+1, j-w:j+w+1, :].flatten()
            Iy = fyTensor[i-w:i+w+1, j-w:j+w+1, :].flatten()
            It = ftTensor[i-w:i+w+1, j-w:j+w+1, :].flatten()
            b = np.reshape(It, (It.shape[0],1)) # get b here
            A = np.vstack((Ix, Iy)).T # get A here
            # if threshold τ is larger than the smallest eigenvalue of A'A:
            nu = np.zeros((2,1))            
            if np.min(abs(np.linalg.eigvals(np.matmul(A.T, A)))) >= tau:
                nu = np.matmul(np.linalg.pinv(A), b) # get velocity here
            
            u[i,j]=nu[0]
            v[i,j]=nu[1]

    return (u,v)




def funcOpticalFlowWarp(x, flo, device='cpu'):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    grid = grid.to(device)
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
    mask = F.grid_sample(mask, vgrid)

    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    return output*mask





def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2
    TAG_CHAR = np.array([202021.25], np.float32)

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()