import cv2
import numpy as np
import os

sampleDir = "images/"

def aslMatch(binary):
    sampleImages = [image for image in os.listdir(sampleDir) if os.path.isfile(os.path.join(sampleDir, image)) and image[-4:] == ".jpg"]

    for sample in sampleImages:
        path = os.path.join(sampleDir, sample)
        template = cv2.imread(path, 0)
        res = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val >= .6:
            print (sample + " | " + str(max_val))
    print("******************************************")
    return
