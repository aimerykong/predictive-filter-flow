# Learning with Predictive Filter Flow for Vision
This github repository contains a series demonstrations of learning with Predictive Filter Flow (PFF) for 
various vision tasks.
PFF is a framework not only supporting fully/self-supervised learning on images and videos, but also 
providing better interpretability that one is able to track every single pixel's movement and its kernels
in constructing the output.
Here is a list of specific applications(*click the link to visit each webpage*):
1. [**image based application**](https://www.ics.uci.edu/~skong2/pff.html "predictive filter flow"): 
	deblur, denoising, defocus, super-resolution, day-night image tranlsation, etc;
2. [**video based application**](https://www.ics.uci.edu/~skong2/mgpff.html): instance tracking, pose tracking, video transition shot detection, frame interpolation, long-range flow learning, style transfer, etc.



PFF             |  mgPFF
:-------------------------:|:-------------------------:
<img src="https://www.ics.uci.edu/~skong2/image2/pff_icon_mediumSize.png" alt="splash figure" width="350"/>  |  <img src="https://www.ics.uci.edu/~skong2/image2/icon_mgpff_small_soccer.gif" alt="splash figure" width="450"/>
[bibtex](./tmp/pff.bibtex) | [bibtex](./tmp/mgpff.bibtex)


---


## Image Reconstruction with Predictive Filter Flow

For paper, slides and poster, please refer to our [project page](https://www.ics.uci.edu/~skong2/pff.html "predictive filter flow")



We propose a simple, interpretable framework for solving a wide range of image
reconstruction problems such as denoising and deconvolution.  Given a
corrupted input image, the model synthesizes a spatially varying linear filter
which, when applied to the input image, reconstructs the desired output. The
model parameters are learned using supervised or self-supervised training.
We test this model on three tasks: non-uniform motion blur removal,
lossy-compression artifact reduction and single image super resolution.  We
demonstrate that our model substantially outperforms state-of-the-art methods
on all these tasks and is significantly faster than optimization-based
approaches to deconvolution.  Unlike models that directly predict output pixel
values, the predicted filter flow is controllable and interpretable, which we
demonstrate by visualizing the space of predicted filters for different tasks.


**keywords**: inverse problem, spatially-variant blind deconvolution, low-level vision, non-uniform motion blur removal, compression artifact reduction, single image super-resolution, filter flow, interpretable model, per-pixel twist, self-supervised learning, image distribution learning.


If you find anything provided here inspires you, please cite our [arxiv paper](https://arxiv.org/abs/1811.11482) ([hig-resolution draft pdf](https://www.ics.uci.edu/~skong2/slides/kf_ff_arxiv2018.pdf), 44Mb):


<img src="https://www.ics.uci.edu/~skong2/image2/pff_icon_mediumSize.png" alt="splash figure" width="350"/>



---



## Multigrid Predictive Filter Flow

For paper, slides and poster, please refer to our [project page](https://www.ics.uci.edu/~skong2/mgpff.html "multigrid predictive filter flow")



We introduce multigrid Predictive Filter Flow (mgPFF), 
a framework for unsupervised learning on videos.
The mgPFF takes as input a pair of frames and outputs per-pixel filters to warp one frame to the other. 
Compared to optical flow used for warping frames, 
mgPFF is more powerful in modeling sub-pixel movement and dealing with corruption (e.g., motion blur). 
We develop a multigrid coarse-to-fine modeling strategy that avoids the requirement of learning large filters to capture large displacement. 
This allows us to train an extremely compact model (**4.6MB**) which operates in a progressive way over multiple resolutions with shared weights. 
We train mgPFF on unsupervised, 
free-form videos and show that mgPFF is able to not only estimate long-range flow for frame reconstruction and detect video shot transitions, 
but also readily amendable for video object segmentation and pose tracking, 
where it substantially outperforms the published state-of-the-art without bells and whistles. 
Moreover, owing to mgPFF's nature of per-pixel filter prediction, 
we have the unique opportunity to visualize how each pixel is evolving during solving these tasks, 
thus gaining better interpretability.

**keywords**: Unsupervised Learning, Multigrid Computing, Long-Range Flow, Video Segmentation, Instance Tracking, Pose Tracking, Video Shot/Transition Detection, Optical Flow, Filter Flow, Low-level Vision.



If you find anything provided here inspires you, please cite our [arxiv paper](https://arxiv.org/abs/XXXX.XXXX)


dog             |  soccerball
:-------------------------:|:-------------------------:
![](https://www.ics.uci.edu/~skong2/image2/icon_mgpff_small_dog.gif)  |  <img src="https://www.ics.uci.edu/~skong2/image2/icon_mgpff_small_soccer.gif" alt="splash figure" width="350"/>





last update: 04/01/2019

Shu Kong


issues/questions addressed here: 
aimerykong At g-m-a-i-l dot com
