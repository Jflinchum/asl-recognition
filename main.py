#!/usr/bin/env python3

import cv2
import numpy as np
import sys
import os.path
from random import randint
from handTrack import getMask, drawContours
from aslRecog import templateMatch
from util import getCoord, getFontSize
from movement import get_movement_ratio

TRAINING_FRAMES = 100

TEMPLATE_PATH = "images/"
TEMPLATE_SIZE = 500

TEXT_FONT = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
TEXT_PLAIN = cv2.FONT_HERSHEY_PLAIN

C_WHITE = (255, 255, 255)
C_RED = (70, 0, 255)

# Main function
def main():
    video = cv2.VideoCapture(0)
    currentFrame = 0
    captureMode = False

    # Get image parameters
    r, i = video.read()
    width, height, channel = i.shape
    topLeftX, topLeftY = getCoord(14, 8, (width, height))
    botRightX, botRightY = getCoord(70, 39, (width, height))

    # Variables for matching the templates
    matches = []
    matchTimer = 50
    maxMatchTimer = 50

    while (video.isOpened()):
        # Constantly read the new frame of the image
        ret, image = video.read()
        image = cv2.flip(image, 1)
        currentFrame = currentFrame + 1

        # Crop video hand area
        croppedHand = image[topLeftY:botRightY, topLeftX:botRightX]

        # Create a copy of the frame and get the mask of it
        maskedHand = getMask(croppedHand.copy(), currentFrame, TRAINING_FRAMES)

        # are we moving?
        move_ratio = get_movement_ratio(croppedHand)

        # Draw the contours onto the original video frame
        contours = drawContours(croppedHand, maskedHand)

        # Text for if the background subtraction is training
        if currentFrame < TRAINING_FRAMES:
            cv2.putText(image, "Training...", getCoord(7, 7, (width, height)), TEXT_FONT, getFontSize(2, image.shape), C_WHITE, 2)

        # Text for capture mode
        if captureMode:
            cv2.putText(image, "Capture Mode", getCoord(7, 50, (width, height)), TEXT_FONT, getFontSize(2, image.shape), C_RED, 2)

        # Box for where the hand is cropped
        cv2.rectangle(image, (botRightX, botRightY), (topLeftX, topLeftY), (0, 255, 0), 0)
        if matchTimer < maxMatchTimer:
            matchTimer = matchTimer + 1
            # Print out matches on image
            for i in range(0, len(matches)):
                imageName, probability = matches[i]
                cv2.putText(image, imageName.replace(TEMPLATE_PATH, "")[0] + "--" + str(probability), getCoord(75, 7 + i*3, (width, height)), TEXT_FONT, getFontSize(1, image.shape), C_WHITE, 1)

        if move_ratio is not None: #and move_ratio < 1.0:
            cv2.putText(image, "{0:.2f}".format(100.*move_ratio), getCoord(7, 70, (width, height)), TEXT_FONT, getFontSize(1, image.shape), C_WHITE, 1)
            if move_ratio < 0.01:
                cv2.putText(image, "STILL", getCoord(7, 74, (width, height)), TEXT_PLAIN, getFontSize(2, image.shape), C_WHITE, 1)
            else:
                cv2.putText(image, "MOVE", getCoord(7, 74, (width, height)), TEXT_PLAIN, getFontSize(2, image.shape), C_WHITE, 1)
        
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
                captureToFile(key, contours, maskedHand)

        else:
            # If space bar is entered, stop video
            if key == ord(" "):
                video.release()

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
                    crop = cv2.resize(maskedHand[y:y+h, x:x+w], (TEMPLATE_SIZE, TEMPLATE_SIZE))
                    matches = templateMatch(crop)
                    matchTimer = 0

"""
captureToFile - Takes the input key, crops the hand, flips the image for opposite
hand capturing, and saves it to a file in the template path
key - key to name the image after
contour - the contour of the image
maskedHand - the mask of the hand to save
"""
def captureToFile(key, contours, maskedHand):
        # Generate random numbers so there are no file collisions
        rand = randint(0, 100000)
        randFlip = randint(0, 100000)
        
        # Crop the image based on contours
        x, y, w, h = cv2.boundingRect(contours)
        crop = maskedHand[y:y+h, x:x+w]

        # We need a mirror image for left and right handed folks
        flipMask = cv2.flip(crop, 1)
        filename = os.path.join(TEMPLATE_PATH, chr(key) + str(rand) + ".jpg")
        filenameFlip = os.path.join(TEMPLATE_PATH, chr(key) + str(randFlip) + ".jpg")
        
        # Write both to filename
        cv2.imwrite(filename, cv2.resize(crop, (TEMPLATE_SIZE, TEMPLATE_SIZE)))
        cv2.imwrite(filenameFlip, cv2.resize(flipMask, (TEMPLATE_SIZE, TEMPLATE_SIZE)))

        print ("Saved as " + filename) 
        print ("Saved as " + filenameFlip) 

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
