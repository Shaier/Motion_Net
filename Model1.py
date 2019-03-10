# Multiple Inputs
'''
Names:
4D CNN
Motion CNN
movement CNN

'''

'''Idea here: Get the weights, run the new frames on the weights, if the loss<threshhold == class 1
THIS MODEL IS MEANT TO ONLY WORK ON ONE MOVEMENT
'''
# 1st input model
frame1 = Input(shape=(25088,))
hidden1 = Dense(100, activation='relu')(frame1)
hidden1 = Dense(100, activation='relu')(hidden1)
output1 = Dense(25088, activation='linear')(hidden1) #frame 2 is output 1

# 2nd input model
frame2 = Input(shape=(25088,))
hidden2 = Dense(100, activation='relu')(frame2)
hidden2 = Dense(100, activation='relu')(hidden2)
output2 = Dense(25088, activation='linear')(hidden2) #frame 3 is output 2

# 3rd input model
frame3 = Input(shape=(25088,))
hidden3 = Dense(100, activation='relu')(frame3)
hidden3 = Dense(100, activation='relu')(hidden3)
output3 = Dense(25088, activation='linear')(hidden3) #frame 4 is output 3

# 4th input model
frame4 = Input(shape=(25088,))
hidden4 = Dense(100, activation='relu')(frame4)
hidden4 = Dense(100, activation='relu')(hidden4)
output4 = Dense(25088, activation='linear')(hidden4) #frame 5 is output 4

# 5th input model
frame5 = Input(shape=(25088,))
hidden5 = Dense(100, activation='relu')(frame5)
hidden5 = Dense(100, activation='relu')(hidden5)
output5 = Dense(25088, activation='linear')(hidden5) #frame 6 is output 5

# 6th input model
frame6 = Input(shape=(25088,))
hidden6 = Dense(100, activation='relu')(frame6)
hidden6 = Dense(100, activation='relu')(hidden6)
output6 = Dense(25088, activation='linear')(hidden6) #frame 7 is output 6

# 7th input model
frame7 = Input(shape=(25088,))
hidden7 = Dense(100, activation='relu')(frame7)
hidden7 = Dense(100, activation='relu')(hidden7)
output7 = Dense(25088, activation='linear')(hidden7) #frame 8 is output 7

# 8th input model
frame8 = Input(shape=(25088,))
hidden8 = Dense(100, activation='relu')(frame8)
hidden8 = Dense(100, activation='relu')(hidden8)
output8 = Dense(25088, activation='linear')(hidden8) #frame 9 is output 8

# 9th input model
frame9 = Input(shape=(25088,))
hidden9 = Dense(100, activation='relu')(frame9)
hidden9 = Dense(100, activation='relu')(hidden9)
output9 = Dense(25088, activation='linear')(hidden9) #frame 10 is output 9

model = Model(inputs=[frame1, frame2, frame3,frame4, frame5, frame6,frame7, frame8, frame9], outputs=[output1, output2, output3,output4, output5, output6, output7,output8, output9])

# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='model.png')


#Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_squared_error'])

#Fit
history = model.fit(x=[train1,train2,train3,train4,train5,train6,train7,train8,train9],
          y=[y1,y2,y3,y4,y5,y6,y7,y8,y9],
          batch_size=100, epochs=100, verbose=1, validation_split=0.2, shuffle=False)

#y=list of Numpy arrays of target (label) data

#Virtualize Training

fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)


'''perhaps for the inputs I need to put x[0], x[1]...
Almost done.
Just need to be able to choose the numbers for the outputs

x: Numpy array of training data (if the model has a single input)==> x= [ [image1],[image2]...]
or list of Numpy arrays (if the model has multiple inputs) ==> x= [ [ [f1],[f2]...[f9] ], [ [f2],[f3]...[f10] ] ]
If input layers in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.  x can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs). If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.  y can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).

model.fit([train_X_hour, train_X_port], [train_Y_hour, train_Y_port] epochs=10 batch_size=1, verbose=2, shuffle=False)

so Maybe
model.fit([train1,train2,train3,train4,train5,train6,train7,train8,train9],[ouput1,output2,output3,ouput4,output5,output6,ouput7,output8,output9],...)
where
train1=[frame1,frame2...frame_N-9]
train2=[frame2,frame3...frame_N-8]
...
train9=[frame9,frame10...frame_N-1] #Note that we go up to N-1 because we need the last frame to be an output

*** Look at them from top to bottom: train1[0],train2[0]...train9[0] is sequence 1.

and
output1=[frame2,frame3...frame_N-8]
output2=[frame3,frame4...frame_N-7]
...
output9=[frame10,frame11...frame_N] #Note that we go up to N because the last frame is an output

'''
'''
Extract features with VGG16'''
vgg = VGG16(weights='imagenet', include_top=False)

#img_path = 'video/img0001.jpg'
'''
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = vgg.predict(x)
features.shape
'''
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
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = vgg.predict(x)
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

#Load the data

#folder with images
image_dir=os.listdir('video')
#create a list to hold the array of pixels of each image
images_array=[]
#place the pixels for each image in the list
for image in image_dir:
    images_array.append(mpimg.imread('video/'+str(image)))


len(images_array)



#imgplot = plt.imshow(img)
train1=[]
train2=[]
train3=[]
train4=[]
train5=[]
train6=[]
train7=[]
train8=[]
train9=[]

train1=images_array[:-9]
train2=images_array[1:-8]
train3=images_array[2:-7]
train4=images_array[3:-6]
train5=images_array[4:-5]
train6=images_array[5:-4]
train7=images_array[6:-3]
train8=images_array[7:-2]
train9=images_array[8:-1]

output1=images_array[1:-8]
output2=images_array[2:-7]
output3=images_array[3:-6]
output4=images_array[4:-5]
output5=images_array[5:-4]
output6=images_array[6:-3]
output7=images_array[7:-2]
output8=images_array[8:-1]
output9=images_array[9:]

def flat(list):
    count=0
    for arr in list:
        newarr=arr.reshape(25088)
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



#get the Loss
hist = pd.DataFrame(history.history)
#hist['epoch'] = history.epoch
#hist.tail()
hist[-1:]

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
3/10- now when the model is running i need a way to compare testing images
2.Inserting 9 images, getting an output of 9 images

3.compare output 1 to input 2, output 2 to input 3... up to output 8 to input 9. (we dont use output 9)

-->MSE is dead simple to implement — but when using it for similarity, we can run into problems. The main one being that large distances
between pixel intensities do not necessarily mean the contents of the images are dramatically different. I’ll provide some proof for that
statement later in this post, but in the meantime, take my word for it.
It’s important to note that a value of 0 for MSE indicates perfect similarity. A value greater than one implies less similarity and will continue to grow as the average difference between pixel intensities increases as well.




'''
