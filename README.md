# Pool

## In this project I will use a new architecture to try and detect movements 
  
THIS MODEL IS MEANT TO ONLY WORK ON ONE MOVEMENT

train1=[frame1,frame2...frame_N-9]
train2=[frame2,frame3...frame_N-8]
train9=[frame9,frame10...frame_N-1] #Note that we go up to N-1 because we need the last frame to be an output

*** Look at them from top to bottom: train1[0],train2[0]...train9[0] is sequence 1.
and
output1=[frame2,frame3...frame_N-8]
output2=[frame3,frame4...frame_N-7]
output9=[frame10,frame11...frame_N] #Note that we go up to N because the last frame is an output

############
Model


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
'''

#############
'''
comparing:
'''
Inserting 9 images, getting an output of 9 images
compare output 1 to input 2, output 2 to input 3... up to output 8 to input 9. (we dont use output 9)

-->MSE is dead simple to implement — but when using it for similarity, we can run into problems. The main one being that large distances
between pixel intensities do not necessarily mean the contents of the images are dramatically different.
It’s important to note that a value of 0 for MSE indicates perfect similarity. A value greater than one implies less similarity and
will continue to grow as the average difference between pixel intensities increases as well.
In order to remedy some of the issues associated with MSE for image comparison, we have the Structural Similarity Index, developed by Wang et al.:

SSIM attempts to model the perceived change in the structural information of the image, whereas MSE is actually estimating the perceived errors.
There is a subtle difference between the two, but the results are dramatic.
Furthermore, the equation in Equation 2 is used to compare two windows (i.e. small sub-samples) rather than the entire image as in MSE.
Doing this leads to a more robust approach that is able to account for changes in the structure of the image, rather than just the perceived change.
Unlike MSE, the SSIM value can vary between -1 and 1, where 1 indicates perfect similarity.


*****I need to use several inputs/outputs because one frame might look the same but the others not and at the end Ill get a bad score
if its a different MOVEMENT
'''

