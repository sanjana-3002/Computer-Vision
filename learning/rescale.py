import cv2 as cv
import numpy as np

# Reading images
img = cv.imread('photos/cat_large.jpg') # this is a larger image hence would be bigger in size and would take more time to process

# Reading Videos
capture = cv.VideoCapture('../Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame, scale=.2)
    
    cv.imshow('Video', frame)
    cv.imshow('Video Resized', frame_resized)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()

# Rescaling images
# we can rescale the image to a smaller size to make it faster to process

def rescaleFrame(frame, scale=0.75): # we use standard scale of 0.75, which means we want to reduce the size of the image by 25%
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale) # frame.shape[1] gives us the new width
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Alternative Method
def changeRes(width,height):
    # Live video
    capture.set(3,width) # capture is the video inputted
    capture.set(4,height)

