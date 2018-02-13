import cv2
import numpy as np
import sys
import os.path
from random import randint
from handTrack import getMask, drawContours
from aslRecog import templateMatch

trainingFrames = 100

# Main function
def main():
    video = cv2.VideoCapture(0)
    currentFrame = 0
    captureMode = False

    while (video.isOpened()):
        # Constantly read the new frame of the image
        ret, image = video.read()

        currentFrame = currentFrame + 1

        # Crop video
        croppedHand = image[100:500, 100:500]

        # Create a copy of the frame and get the mask of it
        maskedHand = getMask(croppedHand.copy(), currentFrame, trainingFrames)

        # Draw the contours onto the original video frame
        drawContours(croppedHand, maskedHand)

        if currentFrame < trainingFrames:
            cv2.putText(image, "Training...", (10, 600), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (255, 255, 255), 2)

        if captureMode:
            cv2.putText(image, "Capture Mode", (10, 50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (70, 0, 255), 2)

        cv2.rectangle(image, (500, 500), (100, 100), (0, 255, 0), 0)

        # Show the frame
        cv2.imshow("video", image)
        cv2.imshow("mask", maskedHand)

        key = cv2.waitKey(10)

        if captureMode: 
            if key == ord(" "):
                captureMode = False
            elif key >= 97 and key <= 122:
                rand = randint(0, 100000)
                filename = chr(key) + str(rand) + ".jpg"
                cv2.imwrite(filename, maskedHand)
                print ("Saved as " + filename) 
        else:
            # If space bar is entered, return to end program
            if key == ord(" "):
                return
            # If r is pressed, reset the frame counter and
            # re-train the background detection
            elif key == ord("r"):
                currentFrame = 0
            elif key == ord("c"):
                captureMode = True
            elif key == ord("t"):
                templateMatch(maskedHand)
            


if __name__ == "__main__":
    main()
