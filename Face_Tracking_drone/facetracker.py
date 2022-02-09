from djitellopy import Tello
from time import sleep
import cv2
import numpy as np

# drone = Tello()
# drone.connect()
# print(drone.get_battery())

def intializeTello():
    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0
    print(myDrone.get_battery())
    myDrone.streamoff()
    myDrone.streamon()
    return myDrone

def findFace(img):
    faceCascade = cv2.CascadeClassifier("D:/drone_programming/haarcascades/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    myFacesListC = []
    myFaceListArea = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        myFacesListC.append([cx, cy])
        myFaceListArea.append(area)

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFacesListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

pid = [0.3, 0.5, 0]
fbRange = [6200, 6800]


def trackFace(tello, center, area, w, pid, pError):
    fb = 0
    x,y = center[0], center[1]
    error = x - w/2
    yaw = pid[0] * error + pid[1] * (error - pError)
    yaw = int(np.clip(yaw, -100,100))
    if area > fbRange[0] and area < fbRange[1]:
        fb = 0
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    if x == 0:
        yaw = 0
        error = 0
    tello.send_rc_control(0, fb, 0, yaw)
    return error

def telloGetFrame(myDrone,w,h):
    myFrame = myDrone.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame, (w, h))
    return img

myDrone = intializeTello()
# cap = cv2.VideoCapture(0)
w=640
h=480
pError = 0
myDrone.takeoff()
while True:
    img = telloGetFrame(myDrone,w,h)
    cv2.imshow("immg", img)
    # ok, img = cap.read()
    img, info = findFace(img)
    pError = trackFace(myDrone, info[0], info[1], w, pid, pError)
    cv2.imshow("MyResult", img)
    if cv2.waitKey(5) & 0xFF == 27:
        myDrone.land()
        break

cv2.destroyAllWindows()