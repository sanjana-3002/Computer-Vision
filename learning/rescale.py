import cv2 as cv
import numpy as np

# Reading images
img = cv.imread('photos/cat_large.jpg') # this is a larger image hence would be bigger in size and would take more time to process

# Rescaling images
# we can rescale the image to a smaller size to make it faster to process

def rescaleFrame(frame, scale=0.75):
    # Images, Videos and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


