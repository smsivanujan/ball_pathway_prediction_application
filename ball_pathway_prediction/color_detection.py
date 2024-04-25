import cv2
from cvzone.ColorModule import ColorFinder

# Create the Color Finder object
myColorFinder = ColorFinder(True)
hsvVals = 'res'

while True:
    # Grab The Image
    img = cv2.imread("Ball.png")
    img = img[0:900, :]

    # Find the Ball Color
    imgColor, mask = myColorFinder.update(img, hsvVals)

    # Display
    img = cv2.resize(img, (0,0), None, 0.7, 0.7)
    cv2.imshow("ImageColor", imgColor)
    cv2.waitKey(50)
