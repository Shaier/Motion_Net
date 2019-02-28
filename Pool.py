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
'''
The convolution function makes use of a for-loop to convolve all the filters over the image. Within each iteration of the for-loop,
two while-loops are used to pass the filter over the image. At each step, the filter is multipled element-wise(*) with a section of the input image.
The result of this element-wise multiplication is then summed to obtain a single value using NumPy’s sum method, and then added with a bias term.
The filt input is initialized using a standard normal distribution and bias is initialized to be a vector of zeros.
After one or two convolutional layers, it is common to reduce the size of the representation produced by the convolutional layer. This reduction in the representation’s size is known as downsampling.
'''
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
'''
The max pooling operation boils down to a for loop and a couple of while loops. The for-loop is used pass through each layer of the input image, and
the while-loops slide the window over every part of the image. At each step, we use NumPy’s max method to obtain the maximum value:
After multiple convolutional layers and downsampling operations, the 3D image representation is converted into a feature vector that is passed into a
Multi-Layer Perceptron, which merely is a neural network with at least three layers. This is referred to as a Fully-Connected Layer.
'''

################################
'''HERE I NEED TO CONNECT ALL OF THE IMAGES INTO ONE VECTOR '''
#############################

'''Fully connected'''
(nf2, dim2, _) = pooled.shape
fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer

'''In this code snippet, we gather the dimensions of the previous layer (number of channels and height/width) then use them to flatten the previous
layer into a fully connected layer. This fully connected layer is proceeded by multiple dense layers of neurons that eventually produce raw predictions:
'''
z = w3.dot(fc) + b3 # first dense layer
z[z<=0] = 0 # pass through ReLU non-linearity
out = w4.dot(z) + b4 # second dense layer

'''Output Layer'''
def softmax(raw_preds):
    '''
    pass raw predictions through softmax activation function
    '''
    out = np.exp(raw_preds) # exponentiate vector of raw predictions
    return out/np.sum(out) # divide the exponentiated vector by its sum. All values in the output sum to 1.

'''Calculating the Loss'''
def categoricalCrossEntropy(probs, label):
    '''
    calculate the categorical cross-entropy loss of the predictions
    '''
    return -np.sum(label * np.log(probs)) # Multiply the desired output label by the log of the prediction, then sum all values in the vector

'''This about wraps up the operations that compose a convolutional neural network. Let us join these operations to construct the CNN.'''


'''example'''
'''Step 1: Getting the Data'''

'''The MNIST handwritten digit training and test data can be obtained here. The files store image and label data as tensors, so the files must be read
 through their bytestream. We define two helper methods to perform the extraction:'''



def extract_data(filename, num_images, IMAGE_WIDTH):
    '''
    Extract images by reading the file bytestream. Reshape the read values into a 3D matrix of dimensions [m, h, w], where m
    is the number of training examples.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data

def extract_labels(filename, num_images):
    '''
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    '''
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

'''Step 2: Initialize parameters'''
def initializeFilter(size, scale = 1.0):
    '''
    Initialize filter using a normal distribution with and a
    standard deviation inversely proportional the square root of the number of units
    '''
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    '''
    Initialize weights with a random normal distribution
    '''
    return np.random.standard_normal(size=size) * 0.01
