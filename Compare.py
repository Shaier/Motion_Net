'''Comparing images'''

'''
Inserting 9 images, getting an output of 9 images
compare output 1 to input 2, output 2 to input 3... up to output 8 to input 9. (we dont use output 9)

-->MSE is dead simple to implement — but when using it for similarity, we can run into problems. The main one being that large distances
between pixel intensities do not necessarily mean the contents of the images are dramatically different.
It’s important to note that a value of 0 for MSE indicates perfect similarity. A value greater than one implies less similarity and
will continue to grow as the average difference between pixel intensities increases as well.
In order to remedy some of the issues associated with MSE for image comparison, we have the Structural Similarity Index, developed by Wang et al.:

SSIM attempts to model the perceived change in the structural information of the image, whereas MSE is actually estimating the perceived errors.
There is a subtle difference between the two, but the results are dramatic.
Furthermore, the equation in Equation 2 is used to compare two windows (i.e. small sub-samples) rather than the entire image as in MSE.
Doing this leads to a more robust approach that is able to account for changes in the structure of the image, rather than just the perceived change.
Unlike MSE, the SSIM value can vary between -1 and 1, where 1 indicates perfect similarity.


*****I need to use several inputs/outputs because one frame might look the same but the others not and at the end Ill get a bad score
if its a different MOVEMENT
'''


!git clone https://github.com/keras-team/keras-contrib
!git clone https://github.com/keras-team/keras-contrib/tree/master/keras_contrib

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


#changing the testing images type so they can be compared

output_images[0]=output_images[0].astype('float32')
output_images[1]=output_images[1].astype('float32')
output_images[2]=output_images[2].astype('float32')
output_images[3]=output_images[3].astype('float32')
output_images[4]=output_images[4].astype('float32')
output_images[5]=output_images[5].astype('float32')
output_images[6]=output_images[6].astype('float32')
output_images[7]=output_images[7].astype('float32')
output_images[8]=output_images[8].astype('float32')

test_list=swim[1:10]

frame1=test_list[0].astype('float32')
frame2=test_list[1].astype('float32')
frame3=test_list[2].astype('float32')
frame4=test_list[3].astype('float32')
frame5=test_list[4].astype('float32')
frame6=test_list[5].astype('float32')
frame7=test_list[6].astype('float32')
frame8=test_list[7].astype('float32')
frame9=test_list[8].astype('float32')


# initialize the figure
fig = plt.figure("Images")
images = ("frame1", frame1), ("frame2", frame2), ("frame3", frame3), ("frame4", frame4), ("frame5", frame5), ("frame6", frame6), ("frame7", frame7), ("frame8", frame8), ("frame9", frame9)


# loop over the images
for (i, (name, image)) in enumerate(images):
	# show the image
	ax = fig.add_subplot(1, 9, i + 1)
	ax.set_title(name)
	plt.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")

#show the figure
plt.show()

#WE NEED TO COMPARE OUTPUT 1 (a1) TO INPUT 2 (images_array2[1] or frame2), OUTPUT 2 (a2) TO INPUT 3 (images_array2[2] or frame3)...
''''''
# compare the images
compare_images(output_images[0], frame2, "output 1 vs input 2")
compare_images(output_images[1], frame3, "output 2 vs input 3")
compare_images(output_images[2], frame4, "output 3 vs input 4")
compare_images(output_images[3], frame5, "output 4 vs input 5")
compare_images(output_images[4], frame6, "output 5 vs input 6")
compare_images(output_images[5], frame7, "output 6 vs input 7")
compare_images(output_images[6], frame8, "output 7 vs input 8")
compare_images(output_images[7], frame9, "output 8 vs input 9")
