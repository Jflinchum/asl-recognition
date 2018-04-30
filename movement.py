
import cv2
import numpy as np
import handTrack

MOVEMENT_BINARY_LOWER = 40
MOVEMENT_BINARY_UPPER = 255
MOVEMENT_THRESHOLD = 0.01

MOVE_REQUIRED = 5
STILL_REQUIRED = 5
MAX_MOVE_RECALL = MOVE_REQUIRED + STILL_REQUIRED

prev_frame = None
move_ratios = []

def get_movement_ratio(frame):
    """
    Get the movement ratio between the current frame and the one prior.
    frame - The current frame of video
    """

    global prev_frame
    change = None
    frame_c = frame.copy()

    # Only detect if we have a previous frame.
    # This will only not pass on the first frame.
    if prev_frame is not None:
        # Get the difference between frames
        diff = cv2.absdiff(prev_frame, frame_c)
        # Convert to grayscale for intensity only
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Only consider pixels with more than MOVEMENT_BINARY_LOWER
        # levels of intensity change as an actual pixel change
        __, diff = cv2.threshold(diff, MOVEMENT_BINARY_LOWER, MOVEMENT_BINARY_UPPER, cv2.THRESH_BINARY)

        # find number of pixels that are considered a significant change
        # and use that to determine the percent of pixels changed in the frame
        change = (cv2.countNonZero(diff) / float(diff.size))

    # Save the frame as the previous frame
    prev_frame = frame_c
    return change

def should_detect(move_ratio):
    """
    Make a robust detection for whether we should detect the hand sign based on movement in the image.
    move_ratio - from the get_movement_ratio function
    """

    global move_ratios

    # should really only happen on the first frame.
    # just ignore it if that is the case
    if move_ratio is None:
        return False

    move_ratios.append(move_ratio)

    # Limit the previous move ratios remembered to MAX_MOVE_RECALL
    if len(move_ratios) > MAX_MOVE_RECALL:
        move_ratios = move_ratios[-MAX_MOVE_RECALL:]

    # If we don't have a full list, we're still in the first MAX_MOVE_RECALL frames.
    # In this scenario, we don't want to detect.
    if len(move_ratios) < MAX_MOVE_RECALL:
        return False

    # Make sure the first MOVE_REQUIRED are above the threshold
    # and the last STILL_REQUIRED are below the threshold
    should = True
    for move in move_ratios[:MOVE_REQUIRED]:
        if move < MOVEMENT_THRESHOLD:
            should = False
    for move in move_ratios[-STILL_REQUIRED:]:
        if move > MOVEMENT_THRESHOLD:
            should = False

    return should
