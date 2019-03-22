'''Comparing images'''

'''
3/10- now when the model is running i need a way to compare testing images
2.Inserting 9 images, getting an output of 9 images

3.compare output 1 to input 2, output 2 to input 3... up to output 8 to input 9. (we dont use output 9)

-->MSE is dead simple to implement — but when using it for similarity, we can run into problems. The main one being that large distances
between pixel intensities do not necessarily mean the contents of the images are dramatically different.
It’s important to note that a value of 0 for MSE indicates perfect similarity. A value greater than one implies less similarity and will continue to grow as the average difference between pixel intensities increases as well.

In order to remedy some of the issues associated with MSE for image comparison, we have the Structural Similarity Index, developed by Wang et al.:

SSIM attempts to model the perceived change in the structural information of the image, whereas MSE is actually estimating the perceived errors. There is a subtle difference between the two, but the results are dramatic.
Furthermore, the equation in Equation 2 is used to compare two windows (i.e. small sub-samples) rather than the entire image as in MSE. Doing this leads to a more robust approach that is able to account for changes in the structure of the image, rather than just the perceived change.
The parameters to Equation 2 include the (x, y) location of the N x N window in each image, the mean of the pixel intensities in the x and y direction, the variance of intensities in the x and y direction, along with the covariance.
Unlike MSE, the SSIM value can vary between -1 and 1, where 1 indicates perfect similarity.

then use ssim loss to compare
perhaps use 96x96, get edges, run model1, train it. then compare 9 images using MSSIM loss function


3/22
take 9 diff images (make sure you have 2 of each - frame 1 and frame 2)
compare the similarities

get a sequence of another MOVEMENT
predict
get the SSIM
add them all
compare that score to the SSIM from a correct sequence
'''

#Not surpassingly, the original image is identical to itself, with a value of 0.0 for MSE and 1.0 for SSIM. Remember, as
#the MSE increases the images are less similar, as opposed to the SSIM where smaller values indicate less similarity.


# import the necessary packages
from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2



#Loss function
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])

	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

	# show the images
	plt.show()


# load the images -- the original, the original + contrast,
# and the original + photoshop
original = cv2.imread("images/jp_gates_original.png")
contrast = cv2.imread("images/jp_gates_contrast.png")
shopped = cv2.imread("images/jp_gates_photoshopped.png")

# convert the images to grayscale
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)


# initialize the figure
fig = plt.figure("Images")
images = ("Original", original), ("Contrast", contrast), ("Photoshopped", shopped)

# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")

# show the figure
plt.show()

# compare the images
compare_images(original, original, "Original vs. Original")
compare_images(original, contrast, "Original vs. Contrast")
compare_images(original, shopped, "Original vs. Photoshopped")

!git clone https://github.com/keras-team/keras-contrib
!git clone https://github.com/keras-team/keras-contrib/tree/master/keras_contrib
