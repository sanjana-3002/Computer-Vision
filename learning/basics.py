import cv2 as cv

img = cv.imread('photos/cat_large.jpg')
cv.imshow('Cat', img)

# Converting to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
# it is used to convert the image into grayscale, mainly used in object or edge detection

# Blur
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)
# it is used to blur the image, mainly used in edge detection

# Edge Cascade
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)
# it is used to detect edges in the image, it uses the canny edge detection algorithm

# Dilating the image
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow('Dilated', dilated)
# it is used to dilate the edges in the image, it uses the dilate function
# It thickens the edges in the image, it is used to make the edges more visible

# Eroding the image
eroded = cv.erode(dilated, (7,7), iterations=3)
cv.imshow('Eroded', eroded)