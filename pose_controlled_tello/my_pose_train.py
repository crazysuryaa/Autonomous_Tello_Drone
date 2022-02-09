import csv
import cv2
import numpy as np
import os
import sys
import tqdm

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose


images_in_folder = "my_poses/train"
images_out_folder = "my_poses/images_out_basic"
csv_out_path = "my_poses/csvs_out_basic.csv"


with open(csv_out_path, 'w') as csv_out_file:
    csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    pose_class_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])

    for pose_class_name in pose_class_names:
        print('Bootstrapping ', pose_class_name, file=sys.stderr)

        if not os.path.exists(os.path.join(images_out_folder, pose_class_name)):
            os.makedirs(os.path.join(images_out_folder, pose_class_name))

        image_names = sorted([
            n for n in os.listdir(os.path.join(images_in_folder, pose_class_name))
            if not n.startswith('.')])
        for image_name in tqdm.tqdm(image_names, position=0):
            # Load image.
            input_frame = cv2.imread(os.path.join(images_in_folder, pose_class_name, image_name))
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

            # Initialize fresh pose tracker and run it.
            with mp_pose.Pose() as pose_tracker:
                result = pose_tracker.process(image=input_frame)
                pose_landmarks = result.pose_landmarks

            # Save image with pose prediction (if pose was detected).
            output_frame = input_frame.copy()
            if pose_landmarks is not None:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)

            # Save landmarks.
            if pose_landmarks is not None:
                # Check the number of landmarks and take pose landmarks.
                assert len(pose_landmarks.landmark) == 33, 'Unexpected number of predicted pose landmarks: {}'.format(
                    len(pose_landmarks.landmark))
                pose_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]

                # Map pose landmarks from [0, 1] range to absolute coordinates to get
                # correct aspect ratio.
                frame_height, frame_width = output_frame.shape[:2]
                pose_landmarks *= np.array([frame_width, frame_height, frame_width])

                # Write pose sample to CSV.
                pose_landmarks = np.around(pose_landmarks, 5).flatten().astype(np.str).tolist()
                csv_out_writer.writerow([image_name, pose_class_name] + pose_landmarks)