import cv2
import numpy as np

fgbg = cv2.createBackgroundSubtractorMOG2()
trained = False

def isTrained():
    global trained
    return trained

# Takes an image in order to greyscale, blur, and apply otsu's method
def getMask(img, currentFrame, trainingFrames = 100):
    global trained
    global fgbg
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
    __, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.dilate(img, np.ones((5,5), np.uint8), iterations=1)
    cv2.erode(img, np.ones((5,5), np.uint8), iterations=1)
    cv2.imshow("mask", img)
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

        # Find convex hull and defects
        hull = cv2.convexHull(maxContour, returnPoints = False)
        defects = cv2.convexityDefects(maxContour, hull)
        # Initialize variables
        palmCenter = (0, 0)
        highestValue = 0
        farPoints = 0
        if (defects is not None):
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i,0]
                start = tuple(maxContour[s][0])
                end = tuple(maxContour[e][0])
                far = tuple(maxContour[f][0])
                # Calculate the center of the palm
                palmCenterX = palmCenter[0] + far[0]
                palmCenterY = palmCenter[1] + far[1]
                farPoints += 1
                palmCenter = palmCenterX, palmCenterY
                # Ignore noise
                if (h * .01) > d/256.0:
                    continue
                if far[1] > highestValue:
                    highestValue = far[1]
                cv2.line(img, start, end, [0,255,0], 2)
                cv2.circle(img, far, 8, [150, 0, 150], -1)

            # Averaging the center of the palm
            if (farPoints != 0):
                palmCenterX = int(palmCenter[0]/(farPoints))
                palmCenterY = int(palmCenter[1]/(farPoints))
                palmCenter = palmCenterX, int((palmCenterY+highestValue)/2)
                cv2.circle(img, palmCenter, 8, [255,0,0], -1)
                if highestValue < y+h:
                    h = highestValue - y
        # Drawing the box and contours
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.drawContours(img, [maxContour], 0, (255, 0, 0), 3)
    return maxContour
