import cv2
import numpy as np
import sys
import os.path

def main():
    path = sys.argv[1]
    if (not os.path.isfile(path)):
        print("Could not find file.")

    # Take the argument and open the image using opencv
    img = cv2.imread(sys.argv[1])
    # Make the image grey
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image a little bit
    img = cv2.GaussianBlur(img,(5,5),0)
    # Applied Otsu's Method
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow("image", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
