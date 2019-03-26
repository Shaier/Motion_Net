'''Extract features with VGG16'''
from keras.applications.vgg16 import VGG16

vgg = VGG16(weights='imagenet', include_top=False)

#folder with images
image_dir=os.listdir('salsa_dance_images')
img_path = 'salsa_dance_images/'

#create a list to hold the array of pixels of each image
images_array=[]
count=0
#place the pixels for each image in the list
for image_name in image_dir:
  try:
    location=str(img_path +str(image_name))
    img = image.load_img(location, target_size=(224, 224))
  # convert image to numpy array
    x = image.img_to_array(img)
  # the image is now in an array of shape (3, 224, 224)
  # but we need to expand it to (1, 2, 224, 224) as Keras is expecting a list of images
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
  # extract the features
    features = vgg.predict(x)
  # convert from Numpy to a list of values
    images_array.append(features)
    if count%1000==0:
      print(count)
    count+=1
  except:
    pass


#Save the list to a file

import pickle

#with open('images_array', 'wb') as fp:
#    pickle.dump(images_array, fp)

with open ('images_array', 'rb') as fp:
    images_array = pickle.load(fp)
