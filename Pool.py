def cov_forward(X,W,b, stride=1,padding=1):
    pass
'''X: DxCxHxW dimension, input filter W: NFxCxHFxHW, and bias b: Fx1'''
'''
D is the number of inputs --> number of images

C is the number of image channel --> 3 RGB or 1 Grey
H is the height of image --> *
W is the width of the image --> *

***** F is the number of frames????

NF is the number of filter in the filter map W
HF is the height of the filter, and finally
HW is the width of the filter.
'''
'''Let’s say we have a single image of 1x1x10x10 size and a single filter of 1x1x3x3. We also use stride of 1 and padding of 1.
Then, naively, if we’re going to do convolution operation for our filter on the image, we will loop over the image, and take the dot
 product at each 3x3 location, because our filter size is 3x3. The result is a single 1x1x10x10 image.
-->SO THE SAME HxW. Which is what I need, just with another dimension for the frames
we will have 100 possible locations to do dot product
At every one of those 100 possible location, there exists the 3x3 patch, stretched to 9x1 column vector that we can do our 3x3 convolution on.
So, with im2col, our image dimension now is: 9x100
Basically, create a similar vector of len 9 to multiply each of the 100 locations (which is one filter)
 '''


 '''wh as weight matrix to the hidden layer
bh as bias matrix to the hidden layer
wout as weight matrix to the output layer
bout as bias matrix to the output layer'''

import numpy as np

#Input array
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
y=np.array([[1],[1],[0]])

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
epoch=5000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):

    #Forward Propogation
    hidden_layer_input1=np.dot(X,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)

    #Backpropagation
    E = y-output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

    print (output)



#####################################################
# here we get rid of that added dimension and plot the image
def visualize_cat(model, cat):
    # Keras expects batches of images, so we have to add a dimension to trick it into being nice
    cat_batch = np.expand_dims(cat,axis=0)
    conv_cat = model.predict(cat_batch)
    conv_cat = np.squeeze(conv_cat, axis=0)
    plt.imshow(conv_cat)

# Note: matplot lib is pretty inconsistent with how it plots these weird cat arrays.
# Try running them a couple of times if the output doesn't quite match the blog post results.
def nice_cat_printer(model, cat):
    '''prints the cat as a 2d array'''
    cat_batch = np.expand_dims(cat,axis=0)
    conv_cat2 = model.predict(cat_batch)

    conv_cat2 = np.squeeze(conv_cat2, axis=0)
    conv_cat2 = conv_cat2.reshape(conv_cat2.shape[:2])

    plt.imshow(conv_cat2)




###################################################################
def convolution(image, filt, bias, s=1):
    '''
    Confolves `filt` over `image` using stride `s`
    '''
    (n_f, n_c_f, f, _) = filt.shape # filter dimensions
    n_c, in_dim, _ = image.shape # image dimensions

    out_dim = int((in_dim - f)/s)+1 # calculate output dimensions

    # ensure that the filter dimensions match the dimensions of the input image
    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"

    out = np.zeros((n_f,out_dim,out_dim)) # create the matrix to hold the values of the convolution operation

    # convolve each filter over the image
    for curr_f in range(n_f):
        curr_y = out_y = 0
        # move filter vertically across the image
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            # move filter horizontally across the image
            while curr_x + f <= in_dim:
                # perform the convolution operation and add the bias
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return out

#The filt input is initialized using a standard normal distribution and bias is initialized to be a vector of zeros.
#After one or two convolutional layers, it is common to reduce the size of the representation produced by the convolutional layer. This reduction in the representation’s size is known as downsampling.

def maxpool(image, f=2, s=2):
    ```
    Downsample input `image` using a kernel size of `f` and a stride of `s`
    ```
    n_c, h_prev, w_prev = image.shape

    # calculate output dimensions after the maxpooling operation.
    h = int((h_prev - f)/s)+1
    w = int((w_prev - f)/s)+1

    # create a matrix to hold the values of the maxpooling operation.
    downsampled = np.zeros((n_c, h, w))

    # slide the window over every part of the image using stride s. Take the maximum value at each step.
    for i in range(n_c):
        curr_y = out_y = 0
        # slide the max pooling window vertically across the image
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            # slide the max pooling window horizontally across the image
            while curr_x + f <= w_prev:
                # choose the maximum value within the window at each step and store it to the output matrix
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled
