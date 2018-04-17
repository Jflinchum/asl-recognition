
import cv2
import numpy as np
import handTrack

MOVEMENT_BINARY_LOWER = 40
MOVEMENT_BINARY_UPPER = 255
# MOVEMENT_THRESHOLD = 1.0 # good between 0.5 - 1.0

prev_frame = None

def get_movement_ratio(frame):
    global prev_frame
    change = None
    frame_c = frame.copy()

    if prev_frame is not None:
        diff = cv2.absdiff(prev_frame, frame_c)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        __, diff = cv2.threshold(diff, MOVEMENT_BINARY_LOWER, MOVEMENT_BINARY_UPPER, cv2.THRESH_BINARY)
        change = (cv2.countNonZero(diff) / float(diff.size))

    prev_frame = frame_c
    return change
