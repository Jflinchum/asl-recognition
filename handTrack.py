import cv2
import numpy as np

fgbg = cv2.createBackgroundSubtractorMOG2()
trained = False

# Takes an image in order to greyscale, blur, and apply otsu's method
def getMask(img, currentFrame, trainingFrames = 100):
    global trained
    global fgbg
    # Make the image grey 
    learning = 0
    
    if currentFrame < trainingFrames:
        if trained:
            fgbg = cv2.createBackgroundSubtractorMOG2()
            trained = False
        learning = -1
    else:
        trained = True
    
    fgmask = fgbg.apply(img, learningRate = learning)
        
    # Blur the image a little bit
    # Maybe do medianBlur or bilateralFilter
    img = cv2.GaussianBlur(fgmask, (15, 15), 0)
    # Applied Otsu's Method
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


# Finds the contours of the imgCopy, presumably the mask of img, and 
# draws the contours onto the img
def drawContours(img, imgCopy):
    # Find the contours in the image
    image, contours, hierarchy = cv2.findContours(imgCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    maxContour = contours
    maxArea = 0
    if (contours):
        # Find largest contour area
        for i in contours:
            area = cv2.contourArea(i)
            if (area > maxArea):
                maxContour = i
                maxArea = area

        # Find bounding box of contours
        x, y, w, h = cv2.boundingRect(maxContour)
        hull = cv2.convexHull(maxContour)
        # Drawing the box and contours
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.drawContours(img, [maxContour], 0, (255, 0, 0), 3)
        cv2.drawContours(img, [hull], -1, (255, 0, 0), 3)
    return maxContour
