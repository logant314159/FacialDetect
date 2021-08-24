import sys
import cv2 as cv
import numpy as np

cascade = cv.CascadeClassifier();
cascade.load("faceCascade.xml")

def GetFaceCoords(frame):
    frameArray = np.fromstring(frame, np.uint8)
    frame = cv.imdecode(frameArray, cv.IMREAD_COLOR)

    grayFrame = cv.equalizeHist(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

    return cascade.detectMultiScale(grayFrame)


