'''detect edges:'''
!git clone https://github.com/JalaliLabUCLA/Image-feature-detection-using-Phase-Stretch-Transform
!pip install Mahotas
os.chdir("/content/gdrive/My Drive/Colab Notebooks/Image-feature-detection-using-Phase-Stretch-Transform/Python")
import PST_function
from PST_function import PST
from os.path import join
# [] Need to install mahotas library for morphological operations
import os
import numpy as np
import mahotas as mh
import matplotlib.pylab as plt
from itertools import zip_longest
#import PST_function
#from PST_function import PST

# [] To process high resolution images set
# from PIL import Image
# Image.MAX_IMAGE_PIXELS = 1000000000
# Replace mh.imread by Image.open

# import sys
# [] To input filename using command line argument uncomment ^^^


# Import the original image
input_path = os.getcwd() # This is where the code is running.
filepath = os.path.join(input_path,'../Test_Images/salsa.jpg')  # The images are located in a folder called 'Test_Images' within the root directory from where the code runs.

#filepath = os.path.join(input_path,'../Test_Images/',sys.argv[1])
# [] To input filename using command line argument uncomment ^^^

Image_orig = mh.imread(filepath) # Read the image.
# To convert the color image to grayscale
if Image_orig.ndim ==3:
    Image_orig_grey = mh.colors.rgb2grey(Image_orig)  # Image_orig is color image.
else:
    Image_orig_grey = Image_orig

# Define various
# low-pass filtering (also called localization kernel) parameter
LPF = 0.21 # Gaussian Low Pass Filter
# PST parameters
Phase_strength = 0.48
Warp_strength= 12.14
# Thresholding parameters (for post processing after the edge is computed)
Threshold_min = -1
Threshold_max = 0.0019
# [] Choose to compute the analog or digital edge,
Morph_flag =1 # [] To compute analog edge, set Morph_flag=0 and to compute digital edge, set Morph_flag=1


[Edge, PST_Kernel]= PST(Image_orig_grey, LPF, Phase_strength, Warp_strength, Threshold_min, Threshold_max, Morph_flag)

if Morph_flag ==0:
    Edge = (Edge/np.max(Edge))*3
    # Display results
    def imshow_pair(image_pair, titles=('', ''), figsize=(6, 6), **kwargs):
        fig, axes = plt.subplots(ncols=len(image_pair), figsize=figsize)
        for ax, img, label in zip_longest(axes.ravel(), image_pair, titles, fillvalue=''):
            ax.imshow(img, **kwargs)
            ax.set_title(label)
    # show the original image and detected edges
    print('        Original Image       Edge Detected using PST')
    imshow_pair((Image_orig, Edge), cmap='gray')

    # Save results
    default_directory, filename=filepath.split('./Test_Images/')
    filename, extension = filename.split('.')
    output_path=default_directory+'./Test_Images/'+filename+'_edge.jpg' # Saving the edge map with the extension tiff
    mh.imsave(output_path, Edge)

else:
    Overlay=mh.overlay(Image_orig_grey,Edge)

    # Display results
    def imshow_pair(image_pair, titles=('', ''), figsize=(10, 6), **kwargs):
        fig, axes = plt.subplots(ncols=len(image_pair), figsize=figsize)
        for ax, img, label in zip_longest(axes.ravel(), image_pair, titles, fillvalue=''):
            ax.imshow(img, **kwargs)
            ax.set_title(label)
    # show the original image, detected edges and an overlay of the original image with detected edges
    print('      Original Image            Edge Detected using PST              Overlay')
    imshow_pair((Image_orig, Edge, Overlay), cmap='gray')

    # Save results
    default_directory, filename=filepath.split('./Test_Images/')
    filename, extension = filename.split('.')
    output_path=default_directory+'./Test_Images/'+filename+'_edge.jpg' # Saving the edge map with the extension tiff
    mh.imsave(output_path, Edge.astype(np.uint8)*255)
    #output_path=default_directory+'./Test_Images/'+filename+'_overlay.jpg' # Saving the overlay with the extension tiff
    #mh.imsave(output_path, Overlay)
