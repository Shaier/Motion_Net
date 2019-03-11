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
