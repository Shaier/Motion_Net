def cov_forward(X,W,b, stride=1,padding=1):
    pass
'''X: DxCxHxW dimension, input filter W: NFxCxHFxHW, and bias b: Fx1'''
'''
D is the number of inputs --> number of images

C is the number of image channel --> 3 RGB or 1 Grey
H is the height of image --> *
W is the width of the image --> *

***** F is the number of frames????

NF is the number of filter in the filter map W
HF is the height of the filter, and finally
HW is the width of the filter.
'''
'''Let’s say we have a single image of 1x1x10x10 size and a single filter of 1x1x3x3. We also use stride of 1 and padding of 1.
Then, naively, if we’re going to do convolution operation for our filter on the image, we will loop over the image, and take the dot
 product at each 3x3 location, because our filter size is 3x3. The result is a single 1x1x10x10 image.
-->SO THE SAME HxW. Which is what I need, just with another dimension for the frames
we will have 100 possible locations to do dot product
At every one of those 100 possible location, there exists the 3x3 patch, stretched to 9x1 column vector that we can do our 3x3 convolution on.
So, with im2col, our image dimension now is: 9x100
Basically, create a similar vector of len 9 to multiply each of the 100 locations (which is one filter)
 '''
