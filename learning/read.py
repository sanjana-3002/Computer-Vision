import cv2 as cv
import numpy as np

# Reading images
img = cv.imread('photos/cat_large.jpg') # this is a larger image hence would be bigger in size and would take more time to process
# additionally, if we would have processed the cat.jpg, it would have been faster as it is a smaller image.
# It totally depends on the pixels ( picture image )
cv.imshow('cat', img)
cv.waitKey(0)

# Reading videos
capture = cv.VideoCapture('videos/dog.mp4')

while True:
    isTrue, frame = capture.read() # this will read the video frame by frame
    if isTrue:
        cv.imshow('video', frame)
        if cv.waitKey(20) & 0xFF == ord('d'): # this will wait for 20 milliseconds and if the user presses 'd' key, it will break the loop
            break
    else:
        break
capture.release() # this will release the video capture object
cv.destroyAllWindows() # this will close all the windows that are opened by OpenCV 