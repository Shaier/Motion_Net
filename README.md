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



'''
