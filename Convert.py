'''Converting videos to frames'''
# Program To Read video and Extract Frames
import cv2
import os

os.chdir('groot')
os.getcwd()
FrameCapture('C:\\Users\\sagi\\Desktop\\groot\\groot.mp4','groot','C:\\Users\\sagi\\Desktop\\groot')
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

        # Saves the frames with frame-count
        cv2.imwrite(video_name+"frame%d.jpg" % count, image)
        count += 1


videos_path='salsa_dance_videos/'
vid_dir=os.listdir(videos_path)
images_path='salsa_dance_images/'

for video in vid_dir:
  video=video.split('.')
  video_name=str(video[0])
  video_path=videos_path+video_name +'.mp4'
  #print(video_path)
  FrameCapture(video_path,video_name,images_path)

# Calling the function
#FrameCapture("salsa_dance_videos/bandicam 2019-03-09 08-51-20-717.mp4")
