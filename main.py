import cv2
import numpy as np
import sys
import os.path
from random import randint
from handTrack import getMask, drawContours
from aslRecog import templateMatch
from util import getCoord

trainingFrames = 100
capturePath = "images/"
templateSize = 500

# Main function
def main():
    video = cv2.VideoCapture(0)
    currentFrame = 0
    captureMode = False

    while (video.isOpened()):
        # Constantly read the new frame of the image
        ret, image = video.read()
        
        width, height, channel = image.shape

        currentFrame = currentFrame + 1

        # Crop video
        topLeftX, topLeftY = getCoord(14, 8, (width, height))
        botRightX, botRightY = getCoord(70, 39, (width, height))
        croppedHand = image[topLeftX:botRightX, topLeftY:botRightY]

        # Create a copy of the frame and get the mask of it
        maskedHand = getMask(croppedHand.copy(), currentFrame, trainingFrames)

        # Draw the contours onto the original video frame
        contours = drawContours(croppedHand, maskedHand)

        # Text for if the background subtraction is training
        if currentFrame < trainingFrames:
            cv2.putText(image, "Training...", getCoord(7, 7, (width, height)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (255, 255, 255), 2)

        # Text for capture mode
        if captureMode:
            cv2.putText(image, "Capture Mode", getCoord(75, 7,(width, height)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, (70, 0, 255), 2)

        # Box for where the hand is cropped
        cv2.rectangle(image, (botRightX, botRightY), (topLeftX, topLeftY), (0, 255, 0), 0)

        # Show the frame
        cv2.imshow("video", image)

        # Wait for a key press
        key = cv2.waitKey(10)

        if captureMode: 
            # Cancel capture mode on a space keypress
            if key == ord(" "):
                captureMode = False
            # If the key is lowercase alphabet letter
            elif key >= 97 and key <= 122:
                # Generate random numbers so there are no file collisions
                rand = randint(0, 100000)
                randFlip = randint(0, 100000)
                
                # Crop the image based on contours
                x, y, w, h = cv2.boundingRect(contours)
                crop = maskedHand[y:y+h, x:x+w]

                # We need a mirror image for left and right handed folks
                flipMask = cv2.flip(crop, 1)
                filename = os.path.join(capturePath, chr(key) + str(rand) + ".jpg")
                filenameFlip = os.path.join(capturePath, chr(key) + str(randFlip) + ".jpg")
                
                # Write both to filename
                cv2.imwrite(filename, cv2.resize(crop, (templateSize, templateSize)))
                cv2.imwrite(filenameFlip, cv2.resize(flipMask, (templateSize, templateSize)))

                print ("Saved as " + filename) 
                print ("Saved as " + filenameFlip) 
        else:
            # If space bar is entered, return to end program
            if key == ord(" "):
                return

            # If r is pressed, reset the frame counter and
            # re-train the background detection
            elif key == ord("r"):
                currentFrame = 0
            # Turn on capture mode on c key press
            elif key == ord("c"):
                captureMode = True
            # Attempt to match the hand 
            elif key == ord("t"):
                if (len(contours) > 0):
                    x, y, w, h = cv2.boundingRect(contours)
                    crop = cv2.resize(maskedHand[y:y+h, x:x+w], (templateSize, templateSize))
                    templateMatch(crop)


if __name__ == "__main__":
    main()
