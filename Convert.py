'''Converting videos to frames'''

# Program To Read video
# and Extract Frames
import cv2

# Function to extract frames
def FrameCapture(path):

    # Path to video file
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    while success:

        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()

        # Saves the frames with frame-count
        cv2.imwrite('salsa_dance_images/'+"frame%d.jpg" % count, image)

        count += 1

# Calling the function
FrameCapture("salsa_dance_videos/bandicam 2019-03-09 08-51-20-717.mp4") 
