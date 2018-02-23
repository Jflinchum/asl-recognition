

"""
getCoord: Returns the width and height coordinates given the percentage of
the image size you want
percentX - percentage along x axis
percentY - percentage along y axis
(width, height) - tuple of the total size of the image
returns - tuple formatted as (x, y) coordinates
"""
def getCoord(percentX, percentY, (width, height)):
    size = (int(width*(percentX/100.0)), int(height*(percentY/100.0)))
    return size
