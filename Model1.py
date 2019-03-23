'''Motion Net'''
'''Idea here: Get the weights, run the new frames on the weights, if the loss<threshhold == class 1
THIS MODEL IS MEANT TO ONLY WORK ON ONE MOVEMENT

train1=[frame1,frame2...frame_N-9]
train2=[frame2,frame3...frame_N-8]
train9=[frame9,frame10...frame_N-1] #Note that we go up to N-1 because we need the last frame to be an output

*** Look at them from top to bottom: train1[0],train2[0]...train9[0] is sequence 1.
and
output1=[frame2,frame3...frame_N-8]
output2=[frame3,frame4...frame_N-7]
output9=[frame10,frame11...frame_N] #Note that we go up to N because the last frame is an output



'''
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

os.chdir("/content/gdrive/My Drive/Colab Notebooks")
os.getcwd()

#Extract Edges:
'''
.
.
.
'''

#Load the data

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


from keras.callbacks import ReduceLROnPlateau
#Early Stop
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


#get the Loss
hist = pd.DataFrame(history.history)
#hist['epoch'] = history.epoch
#hist.tail()
hist[-1:]

#Predicting

#test_list=images_array[1015:1024]
test_list=images_array[1:10]
len(test_list)

flat(test_list)
test_list[1]=np.expand_dims(test_list[1],axis=0)
test_list[2]=np.expand_dims(test_list[2],axis=0)
test_list[3]=np.expand_dims(test_list[3],axis=0)
test_list[4]=np.expand_dims(test_list[4],axis=0)
test_list[5]=np.expand_dims(test_list[5],axis=0)
test_list[6]=np.expand_dims(test_list[6],axis=0)
test_list[7]=np.expand_dims(test_list[7],axis=0)
test_list[8]=np.expand_dims(test_list[8],axis=0)
test_list[0]=np.expand_dims(test_list[0],axis=0)

(a1,a2,a3,a4,a5,a6,a7,a8,a9)=model.predict([ test_list[0],test_list[1],test_list[2],test_list[3],test_list[4],test_list[5],test_list[6],test_list[7],test_list[8] ])

a1=np.reshape(a1,(224,224))
a2=np.reshape(a2,(224,224))
a3=np.reshape(a4,(224,224))
a4=np.reshape(a4,(224,224))
a5=np.reshape(a5,(224,224))
a6=np.reshape(a6,(224,224))
a7=np.reshape(a7,(224,224))
a8=np.reshape(a8,(224,224))
a9=np.reshape(a9,(224,224))


output_image=np.array([a1,a2,a3,a4,a5,a6,a7,a8,a9])
output_image.shape

#plot an image
plt.imshow(output_image[1,:,:])

#plot a big image
fig = plt.figure(figsize=(18, 18))
plt.imshow(a1,cmap='gray')


#plot the sequence of images
fig = plt.figure(figsize=(20, 10))  # width, height in inches
for i in range(9):
    sub = fig.add_subplot(3, 3, i + 1)
    sub.imshow(output_image[i,:,:], interpolation='nearest')


'''
plan:
get the features from the images using a pre trained model
save them into the arrays of training/output
run the model again

Load in a pre-trained CNN model trained on a large dataset
Freeze parameters (weights) in model’s lower convolutional layers
Add custom classifier with several layers of trainable parameters to model
Train classifier layers on training data available for task
Fine-tune hyperparameters and unfreeze more layers as needed

once you got the features move on to the next stage of using ROI --? Maybe not ROI?
the ROI will select the objects to use the CNN on

Maybe train on the entire image, and then use ROI on test data

### problems:
1.using CNN on the input but trying to predict an output that is the actual image and not features
I'll need to run the CNN on the entire dataset first, then put the arrays in the input/output

2.Inserting 9 images, getting an output of 9 images

3.compare output 1 to input 2, output 2 to input 3... up to output 8 to input 9. (we dont use output 9)

4.custom loss function for multiple regression / multi outputs

5. get more images
'''

'''Used Mixamo to get the videos
I ended up downloading the dae/fbx files
But the programs could not change them
So I eventually used Bandicam to record videos
good thing about mixamo... can change angles...
Trying to convert videos to frames now. then run the model


#
use l2 and l1 regularization - model is not overfitting so I dont think I need regularization

for tomorrow:
V check loss before and after each one
V normalize the data
V Change batch size
grid search for hyperparameters =Tried gridcv- cant use functional api. Now trying talos (once I can do grid search I can use search for H.P)--> Try different activation functions, loss function, optimizer.
V Add dropout layer.
V batch norm- right after FC/CNN but before activation like relu
V check if validation_split doesnt shuffle - nope. it takes 10% of the last data (if its set to 0.1?)
V Is the network size is too small / large?
V increase training size
V change the image from /255
Play with the learning rate
Try initialise weights with different initialization scheme.
perhaps run VGG on edges and then run reg.


DONT FORGET THAT THE POINT IS TO PREDICT AT THE END
ADJUST AS MUCH AS YOU CAN THEN PREDICT
Make prediction with one input?

Problem:
'''


#########################
When you call a model, like this:

logits = model(x_train)

the losses it creates during the forward pass are added to the model.losses attribute:

logits = model(x_train[:64])
print(model.losses)

The tracked losses are first cleared at the start of the model __call__, so you will only see the losses created during this one forward pass. For instance, calling the model repeatedly and then querying losses only displays the latest losses, created during the last call:

logits = model(x_train[:64])
logits = model(x_train[64: 128])
logits = model(x_train[128: 192])
print(model.losses)


##############################
'''PREDICT'''
test_list=images_array[1001:1010]
flat(test_list) #convert it to (9216,)
test_list[1]
#output is:
#array([....]) --> notice that there's only one brackets inside the array. WE NEED 2: array( [ [] ] )
#convert them to (1, 9216) from (9216,)
test_list[1]=np.expand_dims(test_list[1],axis=0)
test_list[2]=np.expand_dims(test_list[2],axis=0)
test_list[3]=np.expand_dims(test_list[3],axis=0)
test_list[4]=np.expand_dims(test_list[4],axis=0)
test_list[5]=np.expand_dims(test_list[5],axis=0)
test_list[6]=np.expand_dims(test_list[6],axis=0)
test_list[7]=np.expand_dims(test_list[7],axis=0)
test_list[8]=np.expand_dims(test_list[8],axis=0)
test_list[0]=np.expand_dims(test_list[0],axis=0)

(a1,a2,a3,a4,a5,a6,a7,a8,a9)=model.predict([ test_list[0],test_list[1],test_list[2],test_list[3],test_list[4],test_list[5],test_list[6],test_list[7],test_list[8] ])
#Notice that test_list[i] for all i are all in a list

a1
#output is:
#array([[
#to plot it we need to reshape:
a1=np.reshape(a1,(96,96))
a2=np.reshape(a2,(96,96))
a3=np.reshape(a4,(96,96))
a4=np.reshape(a4,(96,96))
a5=np.reshape(a5,(96,96))
a6=np.reshape(a6,(96,96))
a7=np.reshape(a7,(96,96))
a8=np.reshape(a8,(96,96))
a9=np.reshape(a9,(96,96))


#Plotting
output_image=np.array([a1,a2,a3,a4,a5,a6,a7,a8,a9])
fig = plt.figure(figsize=(20, 10))  # width, height in inches

for i in range(9):
    sub = fig.add_subplot(3, 3, i + 1)
    sub.imshow(output_image[i,:,:], interpolation='nearest')



#################################
3/18

Try using  SSIM as loss function?


3/23
focus on 1 angle to see that it works

1. get a video from 1 angle of a person doing activity -YMCA
2. get another video from the same angle of another person doing the activity
3. get a video of another activity from the same angle with the 1st person
4. get a video of another activity from the same angle with the 1st person
5. compare
