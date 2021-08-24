import sys
import cv2 as cv
import numpy as np

cascade = cv.CascadeClassifier();
cascade.load("faceCascade.xml")

def ParseFrame(image):
    if type(image) == str:
        with open(image, "r") as file:
            frameArray = np.fromstring(file, np.uint8)
        
        return cv.imdecode(frameArray, cv.IMREAD_COLOR)


def GetFaceCoords(frame):
    grayFrame = cv.equalizeHist(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

    return cascade.detectMultiScale(grayFrame)

