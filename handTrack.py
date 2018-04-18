import cv2
import numpy as np

# Takes an image in order to greyscale, blur, and apply otsu's method
def getMask(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    binary = getHSVMask(img)

    # Blur the image a little bit
    binary = cv2.GaussianBlur(binary, (5, 5), 0)
    __, binary = cv2.threshold(binary, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    return binary

def getHSVMask(img): 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    binary = cv2.inRange(hsv, (0, 48, 80), (20, 255, 255))
    return binary

# Finds the contours of the imgCopy, presumably the mask of img, and 
# draws the contours onto the img
def drawContours(img, imgCopy):
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
        cv2.drawContours(image, [maxContour], 0, 255, -1)

        # Find bounding box of contours
        x, y, w, h = cv2.boundingRect(maxContour)
        distTransform = cv2.distanceTransform(image, cv2.DIST_L2, 5)
        centerPixel = np.unravel_index(np.argmax(distTransform, axis=None), distTransform.shape)
        centerPixel = (centerPixel[1], centerPixel[0])

        # Find convex hull and defects
        hull = cv2.convexHull(maxContour, returnPoints = False)
        defects = cv2.convexityDefects(maxContour, hull)
        # Initialize variables
        farPoints = []
        if (defects is not None):
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i,0]
                start = tuple(maxContour[s][0])
                end = tuple(maxContour[e][0])
                far = tuple(maxContour[f][0])

                farPoints.append(far)
                
                # Ignore noise
                if (h * .01) > d/256.0:
                    continue
                cv2.line(img, start, end, [0,255,0], 2)
                cv2.circle(img, far, 8, [150, 0, 150], -1)

        minDist = 10000000
        for point in farPoints:
            distance = dist(point, centerPixel)
            if distance < minDist:
                minDist = int(distance)

        lowerBoxH = centerPixel[1]+minDist+20-y
        boundingBox = (x, y, w, lowerBoxH)

        # Drawing the box and contours
        cv2.circle(img, centerPixel, minDist, [255,0,0], 3)
        cv2.rectangle(img, (x, y), (x+w, y+lowerBoxH), (0, 0, 255), 3)
        cv2.drawContours(img, [maxContour], 0, (255, 0, 0), 3)
        return maxContour, boundingBox
    else:
        return [], None

def dist(x, y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)
