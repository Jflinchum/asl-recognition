
import cv2
import numpy as np
import handTrack

MOVEMENT_BINARY_LOWER = 40
MOVEMENT_BINARY_UPPER = 255
MOVEMENT_THRESHOLD = 0.01

MIN_MOVE_REQUIRED = 5
MIN_STILL_REQUIRED = 5
MAX_MOVE_RECALL = MIN_MOVE_REQUIRED + MIN_STILL_REQUIRED

prev_frame = None
move_ratios = []

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

def should_detect(move_ratio):
    global move_ratios

    if move_ratio is None:
        return False

    move_ratios.append(move_ratio)

    if len(move_ratios) > MAX_MOVE_RECALL:
        move_ratios = move_ratios[-MAX_MOVE_RECALL:]

    if len(move_ratios) < MAX_MOVE_RECALL:
        return False

    should = True

    for move in move_ratios[:MIN_MOVE_REQUIRED]:
        if move < MOVEMENT_THRESHOLD:
            should = False
    for move in move_ratios[-MIN_STILL_REQUIRED:]:
        if move > MOVEMENT_THRESHOLD:
            should = False

    return should
