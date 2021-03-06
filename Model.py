'''Motion Net'''

#Libraries
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import pandas as pd
from keras.layers import Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, Callback
from keras.optimizers import Adam
import keras
import json
from keras.models import load_model
from keras.models import model_from_json
%matplotlib inline

#The first image never loaded correctly so I removed it
shorter_list=images_array[1:] #the list with your edges

#Load the data- arrange in sequences
train1=[]
train2=[]
train3=[]
train4=[]
train5=[]
train6=[]
train7=[]
train8=[]
train9=[]

train1=shorter_list[:-9]
train2=shorter_list[1:-8]
train3=shorter_list[2:-7]
train4=shorter_list[3:-6]
train5=shorter_list[4:-5]
train6=shorter_list[5:-4]
train7=shorter_list[6:-3]
train8=shorter_list[7:-2]
train9=shorter_list[8:-1]

y1=shorter_list[1:-8]
y2=shorter_list[2:-7]
y3=shorter_list[3:-6]
y4=shorter_list[4:-5]
y5=shorter_list[5:-4]
y6=shorter_list[6:-3]
y7=shorter_list[7:-2]
y8=shorter_list[8:-1]
y9=shorter_list[9:]

def flat(list):
    count=0
    for arr in list:
        newarr=arr.reshape(50176)
        list[count]=newarr
        count+=1
flat(y1)
flat(y2)
flat(y3)
flat(y4)
flat(y5)
flat(y6)
flat(y7)
flat(y8)
flat(y9)
flat(train1)
flat(train2)
flat(train3)
flat(train4)
flat(train5)
flat(train6)
flat(train7)
flat(train8)
flat(train9)

# The model
# 1st input model
drop_out=0.2
dense_one=50
dense_two=50
dense_three=10
dense_four=10
pixels=50176

frame1 = Input(shape=(pixels,))
hidden1 = Dense(dense_one)(frame1)
hidden1 = BatchNormalization()(hidden1)
hidden1 = Activation('relu')(hidden1)
hidden1= Dropout(drop_out)(hidden1)
hidden1 = Dense(dense_two)(hidden1)
hidden1 = BatchNormalization()(hidden1)
hidden1 = Activation('relu')(hidden1)
hidden1= Dropout(drop_out)(hidden1)
hidden1 = Dense(dense_three)(hidden1)
hidden1 = BatchNormalization()(hidden1)
hidden1 = Activation('relu')(hidden1)
hidden1= Dropout(drop_out)(hidden1)
hidden1 = Dense(dense_four)(hidden1)
hidden1 = BatchNormalization()(hidden1)
hidden1 = Activation('relu')(hidden1)
hidden1= Dropout(drop_out)(hidden1)
output1 = Dense(pixels, activation='linear')(hidden1) #frame 2 is output 1

# 2nd input model
frame2 = Input(shape=(pixels,))
hidden2 = Dense(dense_one)(frame2)
hidden2 = BatchNormalization()(hidden2)
hidden2 = Activation('relu')(hidden2)
hidden2= Dropout(drop_out)(hidden2)
hidden2 = Dense(dense_two)(hidden2)
hidden2 = BatchNormalization()(hidden2)
hidden2 = Activation('relu')(hidden2)
hidden2= Dropout(drop_out)(hidden2)
hidden2 = Dense(dense_three)(hidden2)
hidden2 = BatchNormalization()(hidden2)
hidden2 = Activation('relu')(hidden2)
hidden2= Dropout(drop_out)(hidden2)
hidden2 = Dense(dense_four)(hidden2)
hidden2 = BatchNormalization()(hidden2)
hidden2 = Activation('relu')(hidden2)
hidden2= Dropout(drop_out)(hidden2)
output2 = Dense(pixels, activation='linear')(hidden2) #frame 3 is output 2

# 3rd input model
frame3 = Input(shape=(pixels,))
hidden3 = Dense(dense_one)(frame3)
hidden3 = BatchNormalization()(hidden3)
hidden3 = Activation('relu')(hidden3)
hidden3= Dropout(drop_out)(hidden3)
hidden3 = Dense(dense_two)(hidden3)
hidden3 = BatchNormalization()(hidden3)
hidden3 = Activation('relu')(hidden3)
hidden3= Dropout(drop_out)(hidden3)
hidden3 = Dense(dense_three)(hidden3)
hidden3 = BatchNormalization()(hidden3)
hidden3 = Activation('relu')(hidden3)
hidden3= Dropout(drop_out)(hidden3)
hidden3 = Dense(dense_four)(hidden3)
hidden3 = BatchNormalization()(hidden3)
hidden3 = Activation('relu')(hidden3)
hidden3= Dropout(drop_out)(hidden3)
output3 = Dense(pixels, activation='linear')(hidden3) #frame 4 is output 3

# 4th input model
frame4 = Input(shape=(pixels,))
hidden4 = Dense(dense_one)(frame3)
hidden4 = BatchNormalization()(hidden4)
hidden4 = Activation('relu')(hidden4)
hidden4= Dropout(drop_out)(hidden4)
hidden4 = Dense(dense_two)(hidden4)
hidden4 = BatchNormalization()(hidden4)
hidden4 = Activation('relu')(hidden4)
hidden4= Dropout(drop_out)(hidden4)
hidden4 = Dense(dense_three)(hidden4)
hidden4 = BatchNormalization()(hidden4)
hidden4 = Activation('relu')(hidden4)
hidden4= Dropout(drop_out)(hidden4)
hidden4 = Dense(dense_four)(hidden4)
hidden4 = BatchNormalization()(hidden4)
hidden4 = Activation('relu')(hidden4)
hidden4= Dropout(drop_out)(hidden4)
output4 = Dense(pixels, activation='linear')(hidden4) #frame 5 is output 4

# 5th input model
frame5 = Input(shape=(pixels,))
hidden5 = Dense(dense_one)(frame5)
hidden5 = BatchNormalization()(hidden5)
hidden5 = Activation('relu')(hidden5)
hidden5= Dropout(drop_out)(hidden5)
hidden5 = Dense(dense_two)(hidden5)
hidden5 = BatchNormalization()(hidden5)
hidden5 = Activation('relu')(hidden5)
hidden5= Dropout(drop_out)(hidden5)
hidden5 = Dense(dense_three)(hidden5)
hidden5 = BatchNormalization()(hidden5)
hidden5 = Activation('relu')(hidden5)
hidden5= Dropout(drop_out)(hidden5)
hidden5 = Dense(dense_four)(hidden5)
hidden5 = BatchNormalization()(hidden5)
hidden5 = Activation('relu')(hidden5)
hidden5= Dropout(drop_out)(hidden5)
output5 = Dense(pixels, activation='linear')(hidden5) #frame 6 is output 5

# 6th input model
frame6 = Input(shape=(pixels,))
hidden6 = Dense(dense_one)(frame6)
hidden6 = BatchNormalization()(hidden6)
hidden6 = Activation('relu')(hidden6)
hidden6= Dropout(drop_out)(hidden6)
hidden6 = Dense(dense_two)(hidden6)
hidden6 = BatchNormalization()(hidden6)
hidden6 = Activation('relu')(hidden6)
hidden6= Dropout(drop_out)(hidden6)
hidden6 = Dense(dense_three)(hidden6)
hidden6 = BatchNormalization()(hidden6)
hidden6 = Activation('relu')(hidden6)
hidden6= Dropout(drop_out)(hidden6)
hidden6 = Dense(dense_four)(hidden6)
hidden6 = BatchNormalization()(hidden6)
hidden6 = Activation('relu')(hidden6)
hidden6= Dropout(drop_out)(hidden6)
output6 = Dense(pixels, activation='linear')(hidden6) #frame 7 is output 6

# 7th input model
frame7 = Input(shape=(pixels,))
hidden7 = Dense(dense_one)(frame7)
hidden7 = BatchNormalization()(hidden7)
hidden7 = Activation('relu')(hidden7)
hidden7= Dropout(drop_out)(hidden7)
hidden7 = Dense(dense_two)(hidden7)
hidden7 = BatchNormalization()(hidden7)
hidden7 = Activation('relu')(hidden7)
hidden7= Dropout(drop_out)(hidden7)
hidden7 = Dense(dense_three)(hidden7)
hidden7 = BatchNormalization()(hidden7)
hidden7 = Activation('relu')(hidden7)
hidden7= Dropout(drop_out)(hidden7)
hidden7 = Dense(dense_four)(hidden7)
hidden7 = BatchNormalization()(hidden7)
hidden7 = Activation('relu')(hidden7)
hidden7= Dropout(drop_out)(hidden7)
output7 = Dense(pixels, activation='linear')(hidden7) #frame 8 is output 7

# 8th input model
frame8 = Input(shape=(pixels,))
hidden8 = Dense(dense_one)(frame8)
hidden1 = BatchNormalization()(hidden8)
hidden8 = Activation('relu')(hidden8)
hidden8= Dropout(drop_out)(hidden8)
hidden8 = Dense(dense_two)(hidden8)
hidden8 = BatchNormalization()(hidden8)
hidden8 = Activation('relu')(hidden8)
hidden8= Dropout(drop_out)(hidden8)
hidden8 = Dense(dense_three)(hidden8)
hidden8 = BatchNormalization()(hidden8)
hidden8 = Activation('relu')(hidden8)
hidden8= Dropout(drop_out)(hidden8)
hidden8 = Dense(dense_four)(hidden8)
hidden8 = BatchNormalization()(hidden8)
hidden8 = Activation('relu')(hidden8)
hidden8= Dropout(drop_out)(hidden8)
output8 = Dense(pixels, activation='linear')(hidden8) #frame 9 is output 8

# 9th input model
frame9 = Input(shape=(pixels,))
hidden9 = Dense(dense_one)(frame9)
hidden9 = BatchNormalization()(hidden9)
hidden9 = Activation('relu')(hidden9)
hidden9= Dropout(drop_out)(hidden9)
hidden9 = Dense(dense_two)(hidden9)
hidden9 = BatchNormalization()(hidden9)
hidden9 = Activation('relu')(hidden9)
hidden9= Dropout(drop_out)(hidden9)
hidden9 = Dense(dense_three)(hidden9)
hidden9 = BatchNormalization()(hidden9)
hidden9 = Activation('relu')(hidden9)
hidden9= Dropout(drop_out)(hidden9)
hidden9 = Dense(dense_four)(hidden9)
hidden9 = BatchNormalization()(hidden9)
hidden9 = Activation('relu')(hidden9)
hidden9= Dropout(drop_out)(hidden9)
output9 = Dense(pixels, activation='linear')(hidden9) #frame 10 is output 9

model = Model(inputs=[frame1, frame2, frame3,frame4, frame5, frame6,frame7, frame8, frame9],
              outputs=[output1, output2, output3,output4, output5, output6, output7,output8, output9])

#Compile the model
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(optimizer=opt, loss='mse', metrics=['mse'])

# summarize layers
print(model.summary())

# plot graph
plot_model(model, to_file='model.png')


#Early Stop
from keras.callbacks import ReduceLROnPlateau
earlystop = EarlyStopping(patience=2)         # Stop training when `val_loss` is no longer improving
        # "no longer improving" being further defined as "for at least 2 epochs"

#Learning Rate Reduction
#We will reduce the learning rate when then accuracy not increase for 2 steps
learning_rate_reduction = ReduceLROnPlateau(monitor='loss',
                                            patience=10,
                                            verbose=1,
                                            min_delta=1e-2,         # "no longer improving" being defined as "no better than 1e-2 less"
                                            factor=0.5,
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]

#load weights
model.load_weights('model_weights.h5')

#Fit
history = model.fit(x=[train1,train2,train3,train4,train5,train6,train7,train8,train9],
          y=[y1,y2,y3,y4,y5,y6,y7,y8,y9], callbacks=callbacks,
          batch_size=10, epochs=90, verbose=1, validation_split=0.1, shuffle=False)


# Save the weights
model.save_weights('model_weights.h5')


#Virtualize Training
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)
ax[1].plot(history.history['loss'], color='b', label="Training loss")
ax[1].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[1])



#Predicting
#test_list=images_array[1015:1024]
test_list=test_array[1:10] #a list with test images

flat(test_list)
test_list[0]=np.expand_dims(test_list[0],axis=0)
test_list[1]=np.expand_dims(test_list[1],axis=0)
test_list[2]=np.expand_dims(test_list[2],axis=0)
test_list[3]=np.expand_dims(test_list[3],axis=0)
test_list[4]=np.expand_dims(test_list[4],axis=0)
test_list[5]=np.expand_dims(test_list[5],axis=0)
test_list[6]=np.expand_dims(test_list[6],axis=0)
test_list[7]=np.expand_dims(test_list[7],axis=0)
test_list[8]=np.expand_dims(test_list[8],axis=0)

#getting the outputs
(output1,output2,output3,output4,output5,output6,output7,output8,output9)=model.predict([ inputs_list[0],inputs_list[1],inputs_list[2],inputs_list[3],inputs_list[4],inputs_list[5],inputs_list[6],inputs_list[7],inputs_list[8] ])

#Reshape images

#outputs
output1=np.reshape(output1,(224,224))
output2=np.reshape(output2,(224,224))
output3=np.reshape(output4,(224,224))
output4=np.reshape(output4,(224,224))
output5=np.reshape(output5,(224,224))
output6=np.reshape(output6,(224,224))
output7=np.reshape(output7,(224,224))
output8=np.reshape(output8,(224,224))
output9=np.reshape(output9,(224,224))

#inputs
inputs_list[0]=np.reshape(output1,(224,224))
inputs_list[1]=np.reshape(output2,(224,224))
inputs_list[2]=np.reshape(output4,(224,224))
inputs_list[3]=np.reshape(output4,(224,224))
inputs_list[4]=np.reshape(output5,(224,224))
inputs_list[5]=np.reshape(output6,(224,224))
inputs_list[6]=np.reshape(output7,(224,224))
inputs_list[7]=np.reshape(output8,(224,224))
inputs_list[8]=np.reshape(output9,(224,224))

output_images=np.array([output1,output2,output3,output4,output5,output6,output7,output8,output9])
output_images.shape

#plot an image
plt.imshow(output_images[1,:,:])

#plot a big image
fig = plt.figure(figsize=(18, 18))
plt.imshow(output1,cmap='gray')


#plot the sequence of images
fig = plt.figure(figsize=(20, 10))  # width, height in inches

for i in range(9):
    sub = fig.add_subplot(3, 3, i + 1)
    sub.imshow(output_images[i,:,:], interpolation='nearest')
