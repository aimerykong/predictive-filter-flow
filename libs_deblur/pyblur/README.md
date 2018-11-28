#Pyblur
##Python image blurring routines.
Pyblur is a collection of simple image blurring routines.<br>
It supports Gaussian, Disk, Box, and Linear Motion Blur Kernels as well as the Point Spread Functions
used in [Convolutional Neural Networks for Direct Text Deblurring](http://www.fit.vutbr.cz/~ihradis/CNN-Deblur/).<br>
Functions receive a PIL image as input, and return another as output.<br>
Kernel sizes can either be specified as input, or randomized.<br>
Finally, there's a RandomizedBlur function that applies a random type of blurring kernel with a random width/strength.

pypi: [https://pypi.python.org/pypi?:action=display&name=pyblur&version=0.2.3](https://pypi.python.org/pypi?:action=display&name=pyblur&version=0.2.3)



##Installation
From Pip: `pip install pyblur`<br>
Or alternatively `git clone` this repo and run locally

##Usage
    from pyblur import *

###Gaussian Blur
Blurs image using a Gaussian Kernel
    
    blurred = GaussianBlur(img, bandwidth)

Randomized kernel bandwidth (between 0.5 and 3.5)

    blurred = GaussianBlur_random(img)

###Defocus (Disk) Blur
Blurs image using a Disk Kernel

	blurred = DefocusBlur(img, kernelsize)

Randomized kernel size (between 3 and 9)

	blurred = DefocusBlur_random(img)


###Box Blur
Blurs image using a Box Kernel

	blurred = BoxBlur(img, kernelsize)

Randomized kernel size (between 3 and 9)

	blurred = BoxBlur_random(img)


###Linear Motion Blur
Blurs image using a Line Kernel

	blurred = LinearMotionBlur(img, dim, angle, linetype)

####Parameters
* `dim` Kernel Size: {3,5,7,9} <br>
* `angle` Angle of the line of motion. Will be floored to the closest one available for the given kernel size. <br>
* `linetype = {left, right, full}` Controls whether the blur kernel will be applied in full or only the left/right halves of it. <br>

Randomized kernel size, angle, and line type

	blurred = LinearMotionBlur_random(img)

### PSF Blur
Blurs image using one of the Point Spread Functions (Kernels) used in:<br>
[Convolutional Neural Networks for Direct Text Deblurring](http://www.fit.vutbr.cz/~ihradis/CNN-Deblur/)

	blurred = PsfBlur(img, psfid)

####Parameters
* `psfid` Id of the Point Spread Function to apply [0, 99] <br>


Randomized kernel size, angle, and line type

	blurred = PsfBlur_random(img)


###Random Blur
Randomly applies one of the supported blur types, with a randomized bandwidth/strenght.

	blurred = RandomizedBlur(img)