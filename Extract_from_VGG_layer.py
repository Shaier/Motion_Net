from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')

# Extract features from an arbitrary intermediate layer
# like the block4 pooling layer in VGG19
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

# load an image and preprocess it
img_path = 'fish.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# get the features
block4_pool_features = model.predict(x)
block4_pool_features.shape


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Sequential
from keras.layers import *
import pylab
model = VGG16(weights='imagenet', include_top=False)

img_path = 'salsa_dance_images/bandicam 2019-03-09 08-51-20-717frame45.jpg'
img = image.load_img(img_path)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
pic=features[0,:,:,21]
pylab.imshow(pic)
pylab.gray()
pylab.show()
