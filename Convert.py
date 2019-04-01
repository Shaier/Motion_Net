'''Program To read video and extract frames'''
import cv2
import os

count=0
fps=2
# Function to extract frames
def FrameCapture(video_path,video_name,images_path):

    # Path to video file
    vidObj = cv2.VideoCapture(video_path)
    # Used as counter variable
    count = 0

    # checks whether frames were extracted
    success = 1

    while success:

        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
        if count % fps ==0: #10 frames per sec
            # Saves the frames with frame-count
            #cv2.imwrite(images_path+'\\'+video_name+"frame%d.jpg" % count, image)
            cv2.imwrite(images_path+'\\'+"%d.jpg" % (count/fps), image)
        count += 1

#Calling the function
#FrameCapture('path\\swim_edge.mp4','swim_edge','path\\swim_edge')

'''
Convert multiple videos
videos_path='videos'
vid_dir=os.listdir(videos_path)
images_path='images'

for video in vid_dir:
  video=video.split('.')
  video_name=str(video[0])
  video_path=videos_path+video_name +'.mp4'
  FrameCapture(video_path,video_name,images_path)
'''
