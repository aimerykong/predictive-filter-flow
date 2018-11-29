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


Please run ``` [task01_deblur.ipynb](https://github.com/aimerykong/predictive-filter-flow/blob/master/task01_deblur.ipynb)``` directly.


Several demos are included as below. 
As for details on the training, demo and code, please go into each demo folder.

1. [demo 1](https://github.com/aimerykong/predictive-filter-flow/tree/master/XXXX) for non-uniform motion blur removal. [**Ready**];
2. [demo 2](https://github.com/aimerykong/predictive-filter-flow/tree/master/XXXX) for JPEG compression artifact reduction [tba];
3. [demo 3](https://github.com/aimerykong/predictive-filter-flow/tree/master/XXXX) for single image super-resolution [tba];
4. [demo 4](https://github.com/aimerykong/predictive-filter-flow/tree/master/XXXX) for visualizing filter flow [tba]


If you find our model/method/dataset useful, please cite our work ([arxiv manuscript](https://arxiv.org/abs/XXXXX)):

    @inproceedings{kong2018PPF,
      title={Image Reconstruction with Predictive Filter Flow},
      author={Kong, Shu and Fowlkes, Charless},
      booktitle={XXXXX},
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
