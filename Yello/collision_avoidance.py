
from core.utils import decode_cfg, load_weights
from core.image import draw_bboxes, preprocess_image, read_image, read_video, Shader
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import tensorflow as tf
import sys
from motrackers.detectors import YOLOv3
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks


from headers import YoloV4Header as Header
from core.model.one_stage.yolov4 import YOLOv4_Tiny as Model
cfg = decode_cfg("cfgs/coco_yolov4_tiny.yaml")
model,evalmodel = Model(cfg,416)



# from headers import YoloV4Header as Header
# from core.model.one_stage.yolov4 import YOLOv4 as Model
# cfg = decode_cfg("cfgs/coco_yolov4.yaml")
# model,evalmodel = Model(cfg,416)
# model.summary()

init_weight_path = cfg['test']['init_weight_path']
if init_weight_path:
    print('Load Weights File From:', init_weight_path)
    load_weights(model, init_weight_path)
else:
    raise SystemExit('init_weight_path is Empty !')



shader = Shader(cfg['yolo']['num_classes'])
names = cfg['yolo']['names']
image_size = cfg['test']['image_size'][0]
# image_size = 416

iou_threshold = cfg["yolo"]["iou_threshold"]
score_threshold = cfg["yolo"]["score_threshold"]
max_outputs = cfg["yolo"]["max_boxes"]
num_classes = cfg["yolo"]["num_classes"]
strides = cfg["yolo"]["strides"]
mask = cfg["yolo"]["mask"]
anchors = cfg["yolo"]["anchors"]




print(image_size)

def preprocess_image(image, size, bboxes=None):
    """
    :param image: RGB, uint8
    :param size:
    :param bboxes:
    :return: RGB, uint8
    """
    iw, ih = size
    h, w, _ = image.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], dtype=np.uint8, fill_value=127)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized

    if bboxes is None:
        return image_paded

    else:
        bboxes = np.asarray(bboxes).astype(np.float32)
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + dw
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + dh

        return image_paded, bboxes



def postprocess_image(image, size, bboxes=None):
    """
    :param image: RGB, uint8
    :param size:
    :param bboxes:
    :return: RGB, uint8
    """
    ih, iw = image.shape[:2]
    w, h = size

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2

    image = image[dh:nh + dh, dw:nw + dw, :]
    image_resized = cv2.resize(image, (w, h))

    if bboxes is None:
        return image_resized
    else:
        bboxes = bboxes.astype(np.float32)
        bboxes[:, [0, 2]] = np.clip((bboxes[:, [0, 2]] - dw) / scale, 0., w)
        bboxes[:, [1, 3]] = np.clip((bboxes[:, [1, 3]] - dh) / scale, 0., h)

        return image_resized, bboxes



def inference(image):
    h, w = image.shape[:2]
    image = preprocess_image(image, (image_size, image_size)).astype(np.float32)
    images = np.expand_dims(image, axis=0)
    images  = images/255.

    tic = time.time()
    pred = model.predict(images)
    bboxes, scores, classes, valid_detections = Header(80, anchors, mask, strides, 10,
                  iou_threshold, score_threshold,inputs = pred)

    # bboxes, scores, classes, valid_detections = evalmodel.predict(images)

    toc = time.time()

    bboxes = bboxes[0][:valid_detections[0]]
    scores = scores[0][:valid_detections[0]]
    classes = classes[0][:valid_detections[0]]
    print(bboxes)
    # bboxes *= image_size
    _, original_boxes = postprocess_image(image, (w, h), bboxes.numpy())

    return (toc - tic) * 1000, original_boxes, scores, classes



cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ms, bboxes, scores, classes = inference(frame)
    image = draw_bboxes(frame, bboxes, scores, classes, names, shader)
    for i,b in enumerate(bboxes):
        # if classes[i] == 3 or classes[i] == 6 or classes[i] == 8:
        
        if scores[i] >= 0.5:
            print(scores[i])
            mid_x = (bboxes[i][1]+bboxes[i][3])/2
            mid_y = (bboxes[i][0]+bboxes[i][2])/2
            print("midx",mid_x,"midy",mid_y)
            # apx_distance = round(((1 - (bboxes[i][3] - bboxes[i][1]))),1)
            apx_distance = (bboxes[i][3] - bboxes[i][1])
            print(apx_distance)
            cv2.putText(image, '{}'.format(apx_distance), (int(mid_x),int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            # if apx_distance <=0.2:
            if apx_distance > image_size-20:
                # if mid_x > 0.3 and mid_x < 0.7:
                cv2.putText(image, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)


    print('Inference Time:', ms, 'ms')
    print('Fps:', 1000/ms)
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
