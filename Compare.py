'''Comparing images'''

'''
3/10- now when the model is running i need a way to compare testing images
2.Inserting 9 images, getting an output of 9 images

3.compare output 1 to input 2, output 2 to input 3... up to output 8 to input 9. (we dont use output 9)

-->MSE is dead simple to implement — but when using it for similarity, we can run into problems. The main one being that large distances
between pixel intensities do not necessarily mean the contents of the images are dramatically different. I’ll provide some proof for that
statement later in this post, but in the meantime, take my word for it.
It’s important to note that a value of 0 for MSE indicates perfect similarity. A value greater than one implies less similarity and will continue to grow as the average difference between pixel intensities increases as well.

In order to remedy some of the issues associated with MSE for image comparison, we have the Structural Similarity Index, developed by Wang et al.:

SSIM attempts to model the perceived change in the structural information of the image, whereas MSE is actually estimating the perceived errors. There is a subtle difference between the two, but the results are dramatic.
Furthermore, the equation in Equation 2 is used to compare two windows (i.e. small sub-samples) rather than the entire image as in MSE. Doing this leads to a more robust approach that is able to account for changes in the structure of the image, rather than just the perceived change.
The parameters to Equation 2 include the (x, y) location of the N x N window in each image, the mean of the pixel intensities in the x and y direction, the variance of intensities in the x and y direction, along with the covariance.
Unlike MSE, the SSIM value can vary between -1 and 1, where 1 indicates perfect similarity.
'''

# import the necessary packages
from skimage.measure import structural_similarity as ssim
