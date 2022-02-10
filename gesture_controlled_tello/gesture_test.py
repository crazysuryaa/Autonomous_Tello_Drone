from gesture_recognition import GestureRecognition , GestureBuffer 
import cv2
from time import sleep
import numpy as np

fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

gesturemodel  =  GestureRecognition()
gesturebuffer = GestureBuffer()
cap = cv2.VideoCapture(0)

while True:
    ok, img = cap.read()
    img = cv2.resize(img,(640, 480))
    debug_image, gesture_id = gesturemodel.recognize(img)
    gesturebuffer.add_gesture(gesture_id)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    cv2.imshow("windows",debug_image)
cv2.destroyAllWindows()
