import cv2
import numpy as np
import sys
import os.path
from random import randint
from handTrack import getMask, drawContours
from aslRecog import aslMatch

trainingFrames = 100

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
        maskedHand = getMask(croppedHand.copy(), currentFrame, trainingFrames)

        # Draw the contours onto the original video frame
        drawContours(croppedHand, maskedHand)

        if currentFrame < trainingFrames:
            cv2.putText(image, "Training...", (10, 600), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow("video", image)
        cv2.imshow("mask", maskedHand)

        key = cv2.waitKey(10)
        # If space bar is entered, return to end program
        if key == ord(" "):
            return
        # If r is pressed, reset the frame counter and
        # re-train the background detection
        elif key == ord("r"):
            currentFrame = 0
        elif key == ord("c"):
            rand = randint(0,100000)
            filename = "hand" + str(rand) + ".jpg"
            cv2.imwrite(filename, maskedHand)
            print ("Saved to " + filename) 
        elif key == ord("t"):
            aslMatch(maskedHand)
            


if __name__ == "__main__":
    main()
