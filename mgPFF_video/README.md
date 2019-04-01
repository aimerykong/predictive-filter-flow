## Multigrid Predictive Filter Flow for Unsupervised Learning on Videos

For paper, slides and poster, please refer to our [project page](https://www.ics.uci.edu/~skong2/mgpff.html "multigrid predictive filter flow")


dog             |  soccerball
:-------------------------:|:-------------------------:
![](https://www.ics.uci.edu/~skong2/image2/icon_mgpff_small_dog.gif)  |  <img src="https://www.ics.uci.edu/~skong2/image2/icon_mgpff_small_soccer.gif" alt="splash figure" width="350"/>





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

    @inproceedings{kong2019mgPPF,
      title={Multigrid Predictive Filter Flow for Unsupervised Learning on Videos},
      author={Kong, Shu and Fowlkes, Charless},
      booktitle={arxiv},
      year={2019}
    }





----

last update: 04/01/2019

Shu Kong


issues/questions addressed here: 
aimerykong At g-m-a-i-l dot com
