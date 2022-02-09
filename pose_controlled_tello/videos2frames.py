import cv2
import os
path = "D:\\drone_programming\\my_poses"
for folder in os.listdir(path):
    print(folder)
    for video in os.listdir(path+"\\"+folder):
        print(video)
        vidcap = cv2.VideoCapture(path+"\\"+folder+"\\"+video)
        success,image = vidcap.read()
        print(success)
        count = 0
        while success:
          cv2.imwrite(path+"\\"+folder+"\\"+"frame%d.jpg" % count, image)     # save frame as JPEG file
          success,image = vidcap.read()
          print('Read a new frame: ', success)
          count += 1
print(count)