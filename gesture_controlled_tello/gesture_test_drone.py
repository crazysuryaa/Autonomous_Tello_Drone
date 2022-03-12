from gesture_recognition import GestureRecognition , GestureBuffer 
from tello_gesture_controller import TelloGestureController
import cv2
from djitellopy import Tello
from time import sleep
import numpy as np

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

def telloGetFrame(myDrone, w, h):
    myFrame = myDrone.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame, (w, h))
    return img

tello = intializeTello()
tello.takeoff()
gesturemodel  =  GestureRecognition()
gestureclassifier = TelloGestureController(tello)
gesturebuffer = GestureBuffer()
# cap = cv2.VideoCapture(0)

while True:
    # ok, img = cap.read()
    img = telloGetFrame(tello, 640, 480)

    debug_image, gesture_id = gesturemodel.recognize(img)
    gesturebuffer.add_gesture(gesture_id)
    gestureclassifier.gesture_control(gesturebuffer)
    if cv2.waitKey(5) & 0xFF == 27:
        tello.land()
        break
    cv2.imshow("windows",debug_image)
cv2.destroyAllWindows()
