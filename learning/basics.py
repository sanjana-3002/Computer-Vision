import cv2 as cv

img = cv.imread('photos/cat_large.jpg')
cv.imshow('Cat', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
# it is used to convert the image into grayscale, mainly used in object or edge detection

