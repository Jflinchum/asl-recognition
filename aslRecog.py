import cv2
import numpy as np
import os

# Uses template matching over a sample directory to show how a binary image
# compares to every image in the directory
def templateMatch(binary, matchFilter, sampleDir):
    sampleImages = [image for image in os.listdir(sampleDir) if os.path.isfile(os.path.join(sampleDir, image)) and image[-4:] == ".jpg"]
    matches = []
    # Iterate through each image in directory
    for sample in sampleImages:
        path = os.path.join(sampleDir, sample)
        template = cv2.imread(path, 0)
        # Skip any images not in the same shape
        if binary.shape != template.shape:
            continue

        # Match the template against the binary image
        res = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the match is greater than the specified ratio, print it
        if max_val > matchFilter:
            finalPath = path.replace(sampleDir, "")[1]
            matches.append((finalPath, max_val))
    matches.sort(key=lambda x: x[1])
    for match in matches:
        print (match)
    print("******************************************")
    return matches
