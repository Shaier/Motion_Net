# Model 2
#Idea here: Get the weights, run the new frames on the weights and predict class 1

# 1st input model
frame1 = Input(shape=(784,1))
hidden1 = Dense(30, activation='relu')(frame1)
hidden1 = Dense(30, activation='relu')(hidden1)

# 2nd input model
frame2 = Input(shape=(784,1))
hidden2 = Dense(30, activation='relu')(frame2)
hidden2 = Dense(30, activation='relu')(hidden2)

# 3rd input model
frame3 = Input(shape=(784,1))
hidden3 = Dense(30, activation='relu')(frame3)
hidden3 = Dense(30, activation='relu')(hidden3)

# 4th input model
frame4 = Input(shape=(784,1))
hidden4 = Dense(30, activation='relu')(frame4)
hidden4 = Dense(30, activation='relu')(hidden4)

# 5th input model
frame5 = Input(shape=(784,1))
hidden5 = Dense(30, activation='relu')(frame5)
hidden5 = Dense(30, activation='relu')(hidden5)

# 6th input model
frame6 = Input(shape=(784,1))
hidden6 = Dense(30, activation='relu')(frame6)
hidden6 = Dense(30, activation='relu')(hidden6)

# 7th input model
frame7 = Input(shape=(784,1))
hidden7 = Dense(30, activation='relu')(frame7)
hidden7 = Dense(30, activation='relu')(hidden7)

# 8th input model
frame8 = Input(shape=(784,1))
hidden8 = Dense(30, activation='relu')(frame8)
hidden8 = Dense(30, activation='relu')(hidden8)

# 9th input model
frame9 = Input(shape=(784,1))
hidden9 = Dense(30, activation='relu')(frame9)
hidden9 = Dense(30, activation='relu')(hidden9)

# merge hidden layers
merge = concatenate([hidden1, hidden2,hidden3, hidden4,hidden5, hidden6,hidden7, hidden8,hidden9])

# interpretation layer
hidden10 = Dense(10, activation='relu')(merge)

# prediction output
output = Dense(1, activation='sigmoid')(hidden10)

model = Model(inputs=[frame1, frame2, frame3,frame4, frame5, frame6,frame7, frame8, frame9], outputs=output)

# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='model2.png')

################
#Load the data

#folder with images
image_dir=os.listdir('video')
#create a list to hold the array of pixels of each image
images_array=[]
#place the pixels for each image in the list
for image in image_dir:
    images_array.append(mpimg.imread('video/'+str(image)))

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
