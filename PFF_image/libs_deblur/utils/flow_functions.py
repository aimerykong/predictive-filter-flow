import torch
import numpy as np
from scipy import signal


class DemoShowFlow():
    def __init__(self, height=151, width=151): # kernel size = [height, width]
        self.colorwheel = 0
        self.makeColorwheel()
        
        truerange = 1
        self.height = height
        self.width = height
        validRange = truerange * 1.04
        
        s2 = np.floor(height/2)
        nx, ny = (width, height)
        xv, yv = np.asarray(range(nx)), np.asarray(range(ny))
        xv, yv = np.meshgrid(xv, yv)

        u = xv*validRange/s2 - validRange
        v = yv*validRange/s2 - validRange
        u /= truerange
        v /= truerange

        self.FlowColorChart = self.computeColor(u, v)

        self.FlowColorChart[int(s2),:,:] = 0
        self.FlowColorChart[:,int(s2),:] = 0
        self.FlowColorChart /= 255.

        
    def filterFlow2UV(self, offsetTensor): # in pytorch tensor format
        kernelSize = int(offsetTensor.size(0)**0.5/2)
        yv, xv = torch.meshgrid([torch.arange(-kernelSize,kernelSize+1), torch.arange(-kernelSize,kernelSize+1)])
        yv, xv = yv.unsqueeze(0).type('torch.FloatTensor'), xv.unsqueeze(0).type('torch.FloatTensor')

        flowMap = offsetTensor.unsqueeze(0)

        yv = yv.contiguous().view(1,-1)
        yv = yv.unsqueeze(-1).unsqueeze(-1)    
        flowMap1 = torch.mul(flowMap,yv)    
        flowMap1 = torch.sum(flowMap1,1)

        xv = xv.contiguous().view(1,-1)
        xv = xv.unsqueeze(-1).unsqueeze(-1)    
        flowMap2 = torch.mul(flowMap,xv)
        flowMap2 = torch.sum(flowMap2,1)

        flowMap = torch.cat([flowMap1,flowMap2],0)
        return flowMap



    def computeColor(self, u, v):        
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
        
        
        