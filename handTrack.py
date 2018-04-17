import cv2
import numpy as np

previousPalms = []
maxPalms = 5

# Takes an image in order to greyscale, blur, and apply otsu's method
def getMask(img, skinTones):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    img = cv2.GaussianBlur(img, (9,9), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    roiHist = getHistogram(skinTones)
    cv2.normalize(roiHist, dst=roiHist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=-1)


    binary = cv2.calcBackProject([hsv], channels=[0, 1], hist=roiHist, ranges=[0, 179, 0, 255], scale=1)

    # Blur the image a little bit
    binary = cv2.medianBlur(binary, 9)
    __, binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations = 1)
    cv2.imshow("binary", binary)
    return binary

def getHistogram(roi):
    histSize = [30, 32]
    ranges = [0, 179, 0, 255]
    channels = [0, 1]
    hist = cv2.calcHist([roi], channels, None, histSize, ranges) 
    return hist

# Finds the contours of the imgCopy, presumably the mask of img, and 
# draws the contours onto the img
def drawContours(img, imgCopy):
    global previousPalms
    # Find the contours in the image
    image, contours, hierarchy = cv2.findContours(imgCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maxContour = contours
    maxArea = -100000
    
    if len(contours) > 0:
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

            # Estimating palm center
            if (farPoints != 0):
                palmCenterX = int(palmCenter[0]/(farPoints))
                palmCenterY = int(palmCenter[1]/(farPoints))
                palmCenter = palmCenterX, int((palmCenterY+highestValue)/2)
                if highestValue < y+h:
                    h = highestValue - y
               
                # Averaging palm center with past palm centers
                if len(previousPalms) >= maxPalms:
                    previousPalms.pop(0)
                previousPalms.append(palmCenter)
                palmCenter = tuple(np.mean(previousPalms, axis=0, dtype=int))

        # Drawing the box and contours
        cv2.circle(img, palmCenter, 8, [255,0,0], -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.drawContours(img, [maxContour], 0, (255, 0, 0), 3)
    return maxContour
