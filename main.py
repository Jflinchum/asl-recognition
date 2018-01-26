import cv2
import numpy as np
import sys
import os.path

def main():
    path = sys.argv[1]
    if (!os.path.isfile(path)):
        print("Could not find file.")

    # Take the argument and open the image using opencv
    img = cv2.imread(sys.argv[1])
    img = img[:,:,0]
    cv2.imshow("image", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
