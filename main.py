import cv2
import numpy as np
import sys
import os.path

# Takes an image in order to greyscale, blur, and apply otsu's method
def getMask(img):
        # Make the image grey
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur the image a little bit
        img = cv2.GaussianBlur(img,(5,5),0)
        # Applied Otsu's Method
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return img

def drawContours(img, imgCopy):
        image, contours, hierarchy = cv2.findContours(imgCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(contours[0])
        cv2.drawContours(img, contours, -1, (0,255,0), 3)
        cv2.drawContours(img, hull, -1, (255, 0, 0), 3)

# Main function
def main():
    cam = False
    # If there are no arguments, capture video from the device's camera
    if len(sys.argv) <= 1:
        cam = True
    if cam:
        # Create the video object
        video = cv2.VideoCapture(0)
        while (video.isOpened()):
            # Constantly read the new frame of the image
            ret, frame = video.read()
            # Create a copy of the frame and get the mask of it
            frameCopy = getMask(frame.copy())
            # Draw the contours onto the original video frame
            drawContours(frame, frameCopy)
            # Show the frame
            cv2.imshow("video", frame)
            key = cv2.waitKey(10)
            # If space bar is entered, return to end program
            if key == 32:
                return
    else:
        path = sys.argv[1]
        if not os.path.isfile(path):
            print("Could not find file.")
        # Take the argument and open the image using opencv
        img = cv2.imread(sys.argv[1])
        imgCopy = getMask(img.copy())
        drawContours(img, imgCopy)
        cv2.imshow("image", img)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
