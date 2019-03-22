'''Converting videos to frames'''
# Program To Read video and Extract Frames
import cv2
import os

os.chdir('C:\\Users\\sagi\\Desktop\\test')
os.getcwd()
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
        if count % 10 ==0: #10 frames per sec
            # Saves the frames with frame-count
            #cv2.imwrite(images_path+'\\'+video_name+"frame%d.jpg" % count, image)
            cv2.imwrite(images_path+'\\'+"%d.jpg" % count, image)
            count += 1

#Calling the function 
FrameCapture('C:\\Users\\sagi\\Desktop\\test\\dance.mp4','dance','C:\\Users\\sagi\\Desktop\\test')


videos_path='salsa_dance_videos/'
vid_dir=os.listdir(videos_path)
images_path='salsa_dance_images/'

#Convert multiple videos
for video in vid_dir:
  video=video.split('.')
  video_name=str(video[0])
  video_path=videos_path+video_name +'.mp4'
  FrameCapture(video_path,video_name,images_path)
