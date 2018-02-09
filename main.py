import cv2
import numpy as np
import sys
import os.path
from handTrack import getMask, drawContours

# Main function
def main():
    video = cv2.VideoCapture(0)
    currentFrame = 0

    while (video.isOpened()):
        # Constantly read the new frame of the image
        ret, image = video.read()

        currentFrame = currentFrame + 1

        # Crop video
        cv2.rectangle(image, (500, 500), (100, 100), (0, 255, 0), 0)
        croppedHand = image[100:500, 100:500]

        # Create a copy of the frame and get the mask of it
        maskedHand = getMask(croppedHand.copy(), currentFrame)

        # Draw the contours onto the original video frame
        drawContours(croppedHand, maskedHand)

        # Show the frame
        cv2.imshow("video", image)
        cv2.imshow("hand", croppedHand)
        cv2.imshow("mask", maskedHand)

        key = cv2.waitKey(10)
        # If space bar is entered, return to end program
        if key == 32:
            return

if __name__ == "__main__":
    main()
