
import cv2

def get_edges(image, mask):
    """
    Run canny edge detection on the hand.
    image - The image to run edge detection on.
    mask - The binary mask to remove all background from the image.
    """

    # Remove hand background to avoid extraneous edges.
    hand = cv2.bitwise_and(image, image, mask = mask)

    # Run Canny edge detection
    edges = cv2.Canny(hand, 100, 200)

    return edges
