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
from edge_detection import get_edges

TEMPLATE_PATH_MASK = "images/mask"
TEMPLATE_PATH_EDGE = "images/edge"
TEMPLATE_SIZE = (200, 200)

TEXT_FONT = cv2.FONT_HERSHEY_TRIPLEX
TEXT_PLAIN = cv2.FONT_HERSHEY_PLAIN

C_WHITE = (255, 255, 255)
C_RED = (70, 0, 255)

# Main function
def main():
    video = cv2.VideoCapture(0)
    currentFrame = 0
    captureMode = False
    translateMode = False

    # Get image parameters
    r, i = video.read()
    height, width, channel = i.shape
    topLeftX, topLeftY = getCoord(8, 14, (width, height))
    botRightX, botRightY = getCoord(39, 70, (width, height))

    # Variables for matching the templates
    matches = {}
    matchTimer = 50
    maxMatchTimer = 50

    # Move Ratio
    previousMoving = False
    moving = False

    while (video.isOpened()):
        # Constantly read the new frame of the image
        ret, image = video.read()
        image = cv2.flip(image, 1)
        currentFrame = currentFrame + 1

        # Crop video hand area
        croppedHand = image[topLeftY:botRightY, topLeftX:botRightX]
        move_check_frame = croppedHand.copy()

        # Create a copy of the frame and get the mask of it
        maskedHand = np.zeros(croppedHand.shape)
            
        maskedHand = getMask(croppedHand.copy())
             
        # Generating canny
        blurredCrop = cv2.bilateralFilter(croppedHand, 9, 75, 75)
        edge_map = get_edges(blurredCrop, maskedHand)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        edge_map = cv2.dilate(edge_map, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations = 2)

        # Draw the contours onto the original video frame
        contours, boundingBox = drawContours(croppedHand, maskedHand)

        # are we moving?
        move_ratio = get_movement_ratio(move_check_frame)

        # Text for capture mode
        if captureMode:
            cv2.putText(image, "Capture Mode", getCoord(7, 80, (width, height)), TEXT_FONT, getFontSize(2, image.shape), C_RED, 2)

        # Text for translate mode
        if translateMode:
            cv2.putText(image, "Translate Mode", getCoord(7, 80, (width, height)), TEXT_FONT, getFontSize(2, image.shape), C_RED, 2)


        # Box for where the hand is cropped
        cv2.rectangle(image, (botRightX, botRightY), (topLeftX, topLeftY), (0, 255, 0), 0)

        if matchTimer < maxMatchTimer:
            matchTimer = matchTimer + 1
            # Print out matches on image
            i = 0
            for match in sorted(matches.items(), key=lambda x: x[1]):
                imageName = match[0]
                probability = match[1]
                cv2.putText(image, imageName + "--" + str(probability), getCoord(50, 7 + i*3, (width, height)), TEXT_FONT, getFontSize(1, image.shape), C_WHITE, 1)
                i += 1

        if move_ratio is not None: #and move_ratio < 1.0:
            cv2.putText(image, "{0:.2f}".format(100.*move_ratio), getCoord(7, 80, (width, height)), TEXT_FONT, getFontSize(1, image.shape), C_WHITE, 1)
            if move_ratio < 0.01:
                previousMoving = moving
                moving = False
                cv2.putText(image, "STILL", getCoord(7, 75, (width, height)), TEXT_PLAIN, getFontSize(2, image.shape), C_WHITE, 1)
            else:
                previousMoving = moving
                moving = True
                cv2.putText(image, "MOVE", getCoord(7, 75, (width, height)), TEXT_PLAIN, getFontSize(2, image.shape), C_WHITE, 1)

        # Attempt to match the hand if the hand was moving and it is now not moving
        if moving == False and previousMoving == True and translateMode:
            if (boundingBox is not None):
                matches = {}
                x, y, w, h = boundingBox
                edge_crop = cv2.resize(edge_map[y:y+h, x:x+w], TEMPLATE_SIZE)
                mask_crop = cv2.resize(maskedHand[y:y+h, x:x+w], TEMPLATE_SIZE)
                matches_mask = templateMatch(mask_crop, .6, TEMPLATE_PATH_MASK)
                matches_edge = templateMatch(edge_crop, .2, TEMPLATE_PATH_EDGE)
                for m in matches_mask:
                    for e in matches_edge:
                        if m[0] == e[0] and m[0] in matches:
                            matches[m[0]] += (m[1] + e[1])
                        else:
                            matches[m[0]] = (m[1] + e[1])
                matchTimer = 0
 
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
                captureToFile(key, boundingBox, maskedHand, TEMPLATE_PATH_MASK)
                captureToFile(key, boundingBox, edge_map, TEMPLATE_PATH_EDGE)

        else:
            # If space bar is entered, stop video
            if key == ord(" "):
                video.release()
            # Turn on capture mode on c key press
            elif key == ord("c"):
                captureMode = True
            elif key == ord("t") and not translateMode:
                translateMode = True
            elif key == ord("t") and translateMode:
                translateMode = False

"""
captureToFile - Takes the input key, crops the hand, flips the image for opposite
hand capturing, and saves it to a file in the template path
key - key to name the image after
boundingBox - the bounding box of the image
maskedHand - the mask of the hand to save
"""
def captureToFile(key, boundingBox, maskedHand, directory):
        # Generate random numbers so there are no file collisions
        rand = randint(0, 100000)
        randFlip = randint(0, 100000)
        
        # Crop the image based on boundingBox
        x, y, w, h = boundingBox 
        crop = maskedHand[y:y+h, x:x+w]

        # We need a mirror image for left and right handed folks
        flipMask = cv2.flip(crop, 1)
        filename = os.path.join(directory, chr(key) + str(rand) + ".jpg")
        filenameFlip = os.path.join(directory, chr(key) + str(randFlip) + ".jpg")
        
        # Write both to filename
        cv2.imwrite(filename, cv2.resize(crop, TEMPLATE_SIZE))
        cv2.imwrite(filenameFlip, cv2.resize(flipMask, TEMPLATE_SIZE))

        print ("Saved as " + filename) 
        print ("Saved as " + filenameFlip) 

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
