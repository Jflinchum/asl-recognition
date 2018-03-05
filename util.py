

"""
getCoord: Returns the width and height coordinates given the percentage of
the image size you want
percentX - percentage along x axis
percentY - percentage along y axis
(width, height) - tuple of the total size of the image
returns - tuple formatted as (x, y) coordinates
"""
def getCoord(percentX, percentY, image_size):
    width, height = image_size # Unpack tuple
    size = (int(width*(percentX/100.0)), int(height*(percentY/100.0)))
    return size

"""
getFontSize: Returns the font size depending on the size of the image
scale - How large the font should be
image_size - tuple of width and height of image
returns - floating point value of font size
"""
def getFontSize(scale, image_size):
    imageScale = 2000.
    totalSize = image_size[0] + image_size[1]
    return scale * (totalSize / imageScale)
