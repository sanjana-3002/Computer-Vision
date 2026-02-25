import cv2 as cv
import numpy as np

# Reading images
img = cv.imread('photos/cat_large.jpg') # this is a larger image hence would be bigger in size and would take more time to process
# additionally, if we would have processed the cat.jpg, it would have been faster as it is a smaller image.
# It totally depends on the pixels ( picture image )
cv.imshow('cat', img)
cv.waitKey(0)

# Reading videos
