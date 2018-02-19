
import cv2

def getGradient(image, mask):

    hand = cv2.bitwise_and(image, image, mask = mask)
    edges = cv2.Canny(hand, 100, 200)

    return edges
