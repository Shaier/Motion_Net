# Model 2

# 1st input model
frame1 = Input(shape=(360, 480, 3))
hidden1 = Dense(30, activation='relu')(frame1)
hidden1 = Dense(30, activation='relu')(hidden1)

# 2nd input model
frame2 = Input(shape=(360, 480, 3))
hidden2 = Dense(30, activation='relu')(frame2)
hidden2 = Dense(30, activation='relu')(hidden2)

# 3rd input model
frame3 = Input(shape=(360, 480, 3))
hidden3 = Dense(30, activation='relu')(frame3)
hidden3 = Dense(30, activation='relu')(hidden3)

# 4th input model
frame4 = Input(shape=(360, 480, 3))
hidden4 = Dense(30, activation='relu')(frame4)
hidden4 = Dense(30, activation='relu')(hidden4)

# 5th input model
frame5 = Input(shape=(360, 480, 3))
hidden5 = Dense(30, activation='relu')(frame5)
hidden5 = Dense(30, activation='relu')(hidden5)

# 6th input model
frame6 = Input(shape=(360, 480, 3))
hidden6 = Dense(30, activation='relu')(frame6)
hidden6 = Dense(30, activation='relu')(hidden6)

# 7th input model
frame7 = Input(shape=(360, 480, 3))
hidden7 = Dense(30, activation='relu')(frame7)
hidden7 = Dense(30, activation='relu')(hidden7)

# 8th input model
frame8 = Input(shape=(360, 480, 3))
hidden8 = Dense(30, activation='relu')(frame8)
hidden8 = Dense(30, activation='relu')(hidden8)

# 9th input model
frame9 = Input(shape=(360, 480, 3))
hidden9 = Dense(30, activation='relu')(frame9)
hidden9 = Dense(30, activation='relu')(hidden9)

# merge hidden layers
merge = concatenate([hidden1, hidden2,hidden3, hidden4,hidden5, hidden6,hidden7, hidden8,hidden9])

# interpretation layer
hidden10 = Dense(10, activation='relu')(merge)

# prediction output
output = Dense(1, activation='sigmoid')(hidden10)

model2 = Model(inputs=[frame1, frame2, frame3,frame4, frame5, frame6,frame7, frame8, frame9], outputs=output)

# summarize layers
print(model2.summary())
# plot graph
plot_model(model2, to_file='model2.png')

#Compile the model
model2.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#Fit
model2.fit(x=[train1,train2,train3,train4,train5,train6,train7,train8,train9],
          y=[],
          batch_size=1, steps_per_epoch=100, epochs=10, verbose=0, validation_split=0.2, shuffle=False)
#x=list of Numpy arrays of training data (x=[ [[f1],[f2]...[f9]], [[f2],[f3]...[f10]] ])
#y=list of Numpy arrays of target (label) data

################
#Load the data

#folder with images
image_dir=os.listdir('video')
#create a list to hold the array of pixels of each image
images_array=[]
#place the pixels for each image in the list
for image in image_dir:
    images_array.append(mpimg.imread('video/'+str(image)))

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



'''
### A problem: using CNN on the input but trying to predict an output that is the actual image and not features
Ill need to run the CNN on the entire dataset first, then put the arrays in the input/output 


'''
