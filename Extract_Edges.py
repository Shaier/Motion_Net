'''Extract Edges:'''

!pip install Mahotas
os.chdir("/content/gdrive/My Drive/Colab Notebooks/Image-feature-detection-using-Phase-Stretch-Transform/Python")
from os.path import join
import PST_function
from PST_function import PST
import mahotas as mh

os.chdir("/content/gdrive/My Drive/Colab Notebooks/Motion_Net")
#folder with images
image_dir=os.listdir('swim_edge')
img_path = 'swim_edge/'

#create a list to hold the array of pixels of each image
swim=[]
count=0
#place the pixels for each image in the list
while len(swim)<len(os.listdir('swim_edge')) and count<500:
  for image_name in os.listdir('swim_edge'):
    image_name=image_name.split('.')
    image_name=str(image_name[0])
    if count==int(image_name):
        swim.append(image_name)
        for image_name in image_dir:
          try:
            location=str(img_path +str(image_name))
            Image_orig = mh.imread(location) # Read the image.
            # To convert the color image to grayscale
            if Image_orig.ndim ==3:
                Image_orig_grey = mh.colors.rgb2grey(Image_orig)  # Image_orig is color image.
                #image.load_img(location, target_size=(224, 224))
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
            else:
                Overlay=mh.overlay(Image_orig_grey,Edge)
                image=Edge.astype(np.uint8)*255
                new_image=mh.imresize(image, (224,224))
          # convert from Numpy to a list of values
            swim.append(new_image)
            if count%50==0:
              print(count)
            count+=1
            if count>2000: #I added this limit because my comp/ google Colab couldn't handle more data than that
              break
          except:
            pass
  count+=1
  print(count)


#Save/ load the list to/from a file
os.chdir("/content/gdrive/My Drive/Colab Notebooks/Motion_Net")
import pickle

#Saving
#with open('swim', 'wb') as fp:
#    pickle.dump(swim, fp)
#Loading
#with open ('images_array3', 'rb') as fp:
#  images_array3 = pickle.load(fp)
