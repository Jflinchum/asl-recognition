
import cv2
import numpy as np

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

def cut_edges_from_binary(binary, edges):
    """
    Add details to the binary image by subtracting the edges.
    binary - Binary image to add detail to
    edges - The edge map for the binary image
    """

    kernel = np.ones((3,3), np.uint8)

    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    inv_edges = cv2.bitwise_not(dilated_edges)
    cut_image = cv2.bitwise_and(binary, binary, mask = inv_edges)

    return cut_image
