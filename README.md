# Motion_Net

## A novel deep learning network I invented to identify movements, rather than objects like most current neural nets do, to detect drowning people 
  
Currently the model works on one movement at a time (though, I believe I could modify it to work on several).  
The architecture currently has 9 inputs (the frames) and 9 outputs (predicted frames). Though, the number of inputs/outputs is arbitrary and can be easily changed based on the complexity of the movement.

The idea behind the architecture:  
Inserting 9 images, getting an output of 9 images.  
Compare output 1 to input 2, output 2 to input 3... up to output 8 to input 9. (we dont use output 9)
The model is trained to predict the next image (we trained it on one movement so imagine the first frame to be a person raising their right hand, second frame the person lowering his right hand... etc.).  
So, given a perfect model, the model will know how the next frame should look like.  
When we insert 9 inputs at the prediction stage, if the model gets 9 images that has different movement (movement that it hasnt seen before), our loss will be high and we'll know that that's the wrong movement.


### The training data is organized as follows:
train1=[frame1,frame2...frame_N-9]  
train2=[frame2,frame3...frame_N-8]  
train9=[frame9,frame10...frame_N-1] 
*Note that we go up to N-1 because we need the last frame to be an output  
  
*** Look at them from top to bottom: train1[0],train2[0]...train9[0] is sequence 1.
  
output1=[frame2,frame3...frame_N-8]  
output2=[frame3,frame4...frame_N-7]  
output9=[frame10,frame11...frame_N] 
*Note that we go up to N because the last frame is an output  


### Obtaining data
I used a website called Mixamo to get the videos and used Bandicam to record the videos from it (since there isnt a simple mp4 download option).  
The good thing about Mixamo is that the dataset you can obtain is limitless. You can change angles, characters, etc. Check it out. 


### Comparing images:
Inserting 9 images, getting an output of 9 images.
compare output 1 to input 2, output 2 to input 3... up to output 8 to input 9. (we dont use output 9).

Mean Squared Error (MSE) is extremely simple to implement — though, when using it for similarities between images we can run into problems. The main one being that large distances
between pixel intensities do not necessarily mean the contents of the images are dramatically different.
It’s important to note that a value of 0 for MSE indicates perfect similarity. A value greater than one implies less similarity and will continue to grow as the average difference between pixel intensities increases as well.
In order to remedy some of the issues associated with MSE for image comparison, I have used Structural Similarity Index (SSIM), developed by Wang et al.  
  
SSIM attempts to model the perceived change in the structural information of the image, whereas MSE is actually estimating the perceived errors. There is a subtle difference between the two, but the results are dramatic.
Doing this leads to a more robust approach that is able to account for changes in the structure of the image, rather than just the perceived change.
Unlike MSE, the SSIM value can vary between -1 and 1, where 1 indicates perfect similarity.  
## The model
