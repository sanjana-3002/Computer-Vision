import cv2 as cv
import numpy as np

# Reading images
img = cv.imread('photos/cat_large.jpg') # this is a larger image hence would be bigger in size and would take more time to process

cv.imshow('cat', img)
cv.waitKey(0)