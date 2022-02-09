import cv2
import pickle
from djitellopy import Tello
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


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


def move_drone(gesture_id):
      forw_back_velocity = 0
      left_right_velocity = 0 
      up_down_velocity = 0
      yaw_velocity= 0
      if gesture_id == 'curl':  # Forward
          forw_back_velocity = 30
      elif  gesture_id == 'none' :  # STOP
          forw_back_velocity = up_down_velocity = \
              left_right_velocity = yaw_velocity = 0
      # if gesture_id == '5':  # Back
      #     forw_back_velocity = -30

      elif gesture_id == 'up':  # UP
          up_down_velocity = 25
      elif gesture_id == 'cross':  # DOWN
          up_down_velocity = -25
          # tello.land()

      elif gesture_id == 'left': # LEFT
          left_right_velocity = 20
      elif gesture_id == 'right': # RIGHT
          left_right_velocity = -20

      print(left_right_velocity, forw_back_velocity,up_down_velocity, yaw_velocity)
      tello.send_rc_control(left_right_velocity, forw_back_velocity,
                                  up_down_velocity, yaw_velocity)


tello = intializeTello()
tello.takeoff()

# cap = cv2.VideoCapture(0)
loaded_knn_model = pickle.load(open('poses_knn', 'rb'))
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  # while cap.isOpened():
  #   success, image = cap.read()
  #   if not success:
  #     print("Ignoring empty camera frame.")
  #     # If loading a video, use 'break' instead of 'continue'.
  #     continue
  while True:
    image = tello.get_frame_read().frame
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    pose_landmarks = results.pose_landmarks


    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    if pose_landmarks is not None:
        assert len(pose_landmarks.landmark) == 33, 'Unexpected number of predicted pose landmarks: {}'.format(
            len(pose_landmarks.landmark))
        pose_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]
        frame_height, frame_width = image.shape[:2]
        pose_landmarks *= np.array([frame_width, frame_height, frame_width])
        pose_landmarks = np.around(pose_landmarks, 5).flatten().astype(np.str)
        ypred = loaded_knn_model.predict(pose_landmarks.reshape(1,-1))[0]
        print(ypred)
        move_drone(ypred)

    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      tello.land()
      break
cv2.destroyAllWindows()
# cap.release()