import cv2 as cv
import numpy as np

img = cv.imread('photos/cat_large.jpg')
cv.imshow('cat', img)

cv.waitKey(0)