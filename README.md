# Learning with Predictive Filter Flow for Vision
This github repository contains a series demonstrations of learning with Predictive Filter Flow (PFF) for 
various vision tasks.
PFF is a framework not only supporting fully/self-supervised learning on images and videos, but also 
providing better interpretability that one is able to track every single pixel's movement and its kernels
in constructing the output.
Here is a list of specific applications:
1. **image based application**: deblur, denoising, defocus, super-resolution, day-night image tranlsation, etc;
2. **video based application**: instance tracking, pose tracking, video transition shot detection, frame interpolation, long-range flow learning, style transfer, etc.








# Image Reconstruction with Predictive Filter Flow


For paper, slides and poster, please refer to our [project page](https://www.ics.uci.edu/~skong2/pff.html "predictive filter flow")



<img src="https://www.ics.uci.edu/~skong2/image2/pff_icon_mediumSize.png" alt="splash figure" width="350"/>

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

The jupyter script provided here is self-contained.
Please run [task01_deblur.ipynb](https://github.com/aimerykong/predictive-filter-flow/blob/master/task01_deblur.ipynb) directly to see how our model performs for non-uniform motion blur removal.
Please go to [this folder](https://github.com/aimerykong/predictive-filter-flow/tree/master/libs_deblur/result/._epoch-445/moderateBlurDataset_fullRes) to have a quick look at more visualizations!
Besides, more demos are on the way.



If you find anything provided here inspires you, please cite our [arxiv paper](https://arxiv.org/abs/1811.11482) ([hig-resolution draft pdf](https://www.ics.uci.edu/~skong2/slides/kf_ff_arxiv2018.pdf), 44Mb):

    @inproceedings{kong2018PPF,
      title={Image Reconstruction with Predictive Filter Flow},
      author={Kong, Shu and Fowlkes, Charless},
      booktitle={arxiv},
      year={2018}
    }



![alt text](https://www.ics.uci.edu/~skong2/image2/pff_demo_motion_deblur.png "visualization")

![alt text](https://www.ics.uci.edu/~skong2/image2/pff_demo_jpeg.png "visualization")

![alt text](https://www.ics.uci.edu/~skong2/image2/pff_demo_SISR.png "visualization")

![alt text](https://www.ics.uci.edu/~skong2/image2/pff_demo_analysisFF.png "visualization")


last update: 10/28/2018

Shu Kong


issues/questions addressed here: 
aimerykong At g-m-a-i-l dot com
