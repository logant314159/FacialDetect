import sys
import cv2 as cv
import numpy as np

cascade = cv.CascadeClassifier();
cascade.load("faceCascade.xml")

def GetFaceCoords(frame):
    grayFrame = cv.equalizeHist(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
    return cascade.detectMultiScale(grayFrame)


def DrawRect(frame, coords):
    for (x, y, w, h) in coords:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)


if "-f" in sys.argv:
    path = sys.argv[2]
    extension = path[-3:]

    if extension in ["png", "jpg"]:
        image = cv.imread(path)
        coords = GetFaceCoords(image)

        DrawRect(image, coords)
        
        cv.imwrite(f"output.{extension}", image)

    elif extension in ["mp4", "avi"]:
        cap = cv.VideoCapture(path)

        width = int(cap.get(3))
        height = int(cap.get(4))

        out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M','J','P','G'), 10, (width, height))

        ret, frame = cap.read()

        while ret:
            coords = GetFaceCoords(frame)

            DrawRect(frame, coords)

            out.write(frame)

            ret, frame = cap.read()

        cap.release()
        out.release()

elif "-w" in sys.argv:
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        coords = GetFaceCoords(frame)
        DrawRect(frame, coords)

        cv.imshow("Webcam", frame)

        k = cv.waitKey(30) & 0xff
        if k == 27: # Press esc to exit.
            break;

    cap.release()

elif "-h" in sys.argv:
    print("\nUsage: FacialDetect.py <option> <path/to/file.mp4>\n")
    print("""OPTIONS:
    \t-h Diplay help.
    \t-f Read from, then write to an image or video.
    \t-w Display live webcam with facial detection active.
    """)
    print("""EXAMPLES:
    \tFacialDetect.py -f image.jpg
    \tFacialDetect.py -f media/video.mp4
    \tFacialDetect.py -w""")

else:
    print("Argument unrecognized. Use the -h flag for examples.")