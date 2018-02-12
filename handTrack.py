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
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img


# Finds the contours of the imgCopy, presumably the mask of img, and 
# draws the contours onto the img
def drawContours(img, imgCopy):
    # Find the contours in the image
    image, contours, hierarchy = cv2.findContours(imgCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    
    if (contours):
        # Find largest contour area
        maxArea = 0
        maxContour = contours[0]
        for i in contours:
            area = cv2.contourArea(i)
            if (area > maxArea):
                maxArea = area
                maxContour = i
        
        # Find bounding box of contours
        rect = cv2.minAreaRect(maxContour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Drawing the box and contours
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
        cv2.drawContours(img, maxContour, -1, (255,0,0), 3)
    return contours
