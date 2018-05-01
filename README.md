# asl-recognition
CSCI-431 Spring 2018 project for recognizing the ASL alphabet.

## Installation
Requires [OpenCV](https://opencv.org/) and [NumPy](http://www.numpy.org/) to run.

## Usage
Running `main.py` will start the main program, which uses the device's camera.
The user starts the application in Normal mode.

## Modes

### Normal Mode
Shows the cropped area of the hand and any points detected on it. This mode does not attempt to translate.
Movement detection is shown to the user under the cropped area, but not used.

* `c` - Change to Capture mode
* `t` - Change to Translate mode
* `<space>` - Quit

### Capture Mode
Records binary and Canny hand data for any alphabetic character.
The user only needs to record either the left-hand or right-hand version, but not both, as it flips the image and saves each orientation.

* `[a-z]` - Save frame data for specified letter
* `<space>` - Change to Normal mode

### Translate Mode
Attempts to translate the shown sign of the hand detected in the cropped area. It begins detection when it goes from moving to still in the motion detection.
It will display the top results for which letter is thinks is shown in ascending order on the right and the translate history at the bottom.

* `t` - Change to Normal mode

## Contributors
* Jonathan Flinchum
* Kathryn France
* Holden Lewis
* Nicole Morken
