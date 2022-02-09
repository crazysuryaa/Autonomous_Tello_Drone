import tensorflow as tf

!pip install opencv-python



from core.losses.iou import GIoU, DIoU, CIoU
WEIGHT_DECAY = 0.  # 5e-4
LEAKY_ALPHA = 0.1

# Header(num_classes = num_classes, anchors = anchors, mask = mask, strides = strides, max_outputs= max_outputs, iou_threshold = iou_threshold, score_threshold = score_threshold,inputs = pred )

def Header(num_classes, anchors, mask, strides,
                 max_outputs, iou_threshold, score_threshold,inputs):
    boxes, objects, classes = [], [], []
    dtype = inputs[0].dtype

    
    for i, logits in enumerate(inputs):

        stride = strides[i]
        anchors = anchors[mask[i]]
        x_shape = tf.shape(logits)
        logits = tf.reshape(logits, (x_shape[0], x_shape[1], x_shape[2], len(anchors), num_classes + 5))

        box_xy, box_wh, obj, cls = tf.split(logits, (2, 2, 1, num_classes), axis=-1)
        box_xy = tf.sigmoid(box_xy)
        obj = tf.sigmoid(obj)
        cls = tf.sigmoid(cls)
        anchors = anchors.astype(np.float32)

        grid_shape = x_shape[1:3]
        # print(grid_shape)
        grid_h, grid_w = grid_shape[0], grid_shape[1]
        # print(grid_h,tf.range(grid_h))
        
        grid = tf.meshgrid(tf.range(grid_w), tf.range(grid_h))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        box_xy = (box_xy + tf.cast(grid, dtype)) * stride
        box_wh = tf.exp(box_wh) * anchors

        box_x1y1 = box_xy - box_wh / 2.
        box_x2y2 = box_xy + box_wh / 2.
        box = tf.concat([box_x1y1, box_x2y2], axis=-1)

        boxes.append(tf.reshape(box, (x_shape[0], -1, 1, 4)))
        objects.append(tf.reshape(obj, (x_shape[0], -1, 1)))
        classes.append(tf.reshape(cls, (x_shape[0], -1, num_classes)))

    boxes = tf.concat(boxes, axis=1)
    objects = tf.concat(objects, axis=1)
    classes = tf.concat(classes, axis=1)

    scores = objects * classes
    boxes, scores, classes, valid = tf.image.combined_non_max_suppression(
        boxes=boxes,
        scores=scores,
        max_output_size_per_class=max_outputs,
        max_total_size=max_outputs,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        clip_boxes=False
    )

    return boxes, scores, classes, valid    



def _broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) - tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) - tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * (box_2[..., 3] - box_2[..., 1])
    return int_area / tf.maximum(box_1_area + box_2_area - int_area, 1e-8)

def DarknetConv2D_BN_Leaky(x,*args, **kwargs):
    without_bias_kwargs = {"use_bias": False}
    without_bias_kwargs.update(kwargs)

    x = DarknetConv2D(x,*args, **without_bias_kwargs)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.ReLU()(x)
    
    
    return x

def RouteGroup(x,ngroups,group_id):
    convs = tf.split(x, num_or_size_splits=ngroups, axis=-1)
    return convs[group_id]
    

def DarknetConv2D(x,*args, **kwargs):
    darknet_conv_kwargs = {"kernel_regularizer": tf.keras.regularizers.l2(WEIGHT_DECAY),
                           "kernel_initializer": tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                           "padding": "valid" if kwargs.get(
                               "strides") == (2, 2) else "same"}
    darknet_conv_kwargs.update(kwargs)

    return tf.keras.layers.Conv2D(*args, **darknet_conv_kwargs)(x)


def YOLOv4_Tiny(cfg,
                input_size=None,
                name=None):
    iou_threshold = cfg["yolo"]["iou_threshold"]
    score_threshold = cfg["yolo"]["score_threshold"]
    max_outputs = cfg["yolo"]["max_boxes"]
    num_classes = cfg["yolo"]["num_classes"]
    strides = cfg["yolo"]["strides"]
    mask = cfg["yolo"]["mask"]
    anchors = cfg["yolo"]["anchors"]

    if input_size is None:
        x = inputs = tf.keras.Input([None, None, 3])
    else:
        x = inputs = tf.keras.Input([input_size, input_size, 3])

    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(x,32, (3, 3), strides=(2, 2))  # 0
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(x,64, (3, 3), strides=(2, 2)) # 1
    x = DarknetConv2D_BN_Leaky(x,64, (3, 3))  # 2

    route = x
    x = RouteGroup(x,2, 1) # 3
    x = DarknetConv2D_BN_Leaky(x,32, (3, 3))  # 4
    route_1 = x
    x = DarknetConv2D_BN_Leaky(x,32, (3, 3))  # 5
    x = tf.keras.layers.Concatenate()([x, route_1])  # 6
    x = DarknetConv2D_BN_Leaky(x,64, (1, 1))  # 7
    x = tf.keras.layers.Concatenate()([route, x])  # 8
    x = tf.keras.layers.MaxPooling2D(2, 2, padding='same')(x)  # 9

    x = DarknetConv2D_BN_Leaky(x,128, (3, 3)) # 10
    route = x
    x = RouteGroup(x,2, 1) # 11
    x = DarknetConv2D_BN_Leaky(x,64, (3, 3))  # 12
    route_1 = x
    x = DarknetConv2D_BN_Leaky(x,64, (3, 3)) # 13
    x = tf.keras.layers.Concatenate()([x, route_1])  # 14
    x = DarknetConv2D_BN_Leaky(x,128, (1, 1)) # 15
    x = tf.keras.layers.Concatenate()([route, x])  # 16
    x = tf.keras.layers.MaxPooling2D(2, 2, padding='same')(x)  # 17

    x = DarknetConv2D_BN_Leaky(x,256, (3, 3))  # 18
    route = x
    x = RouteGroup(x,2, 1)  # 19
    x = DarknetConv2D_BN_Leaky(x,128, (3, 3))  # 20
    route_1 = x
    x = DarknetConv2D_BN_Leaky(x,128, (3, 3)) # 21
    x = tf.keras.layers.Concatenate()([x, route_1])  # 22
    x = x_23 = DarknetConv2D_BN_Leaky(x,256, (1, 1))  # 23
    x = tf.keras.layers.Concatenate()([route, x])  # 24
    x = tf.keras.layers.MaxPooling2D(2, 2, padding='same')(x)  # 25
    x = DarknetConv2D_BN_Leaky(x,512, (3, 3))

    x = DarknetConv2D_BN_Leaky(x,256, (1, 1))

    _x = DarknetConv2D_BN_Leaky(x,512, (3, 3))
    output_0 = DarknetConv2D(_x,len(mask[0]) * (num_classes + 5), (1, 1),name = "out0")

    x = DarknetConv2D_BN_Leaky(x,128, (1, 1))
    x = tf.keras.layers.UpSampling2D(2)(x)
    x = tf.keras.layers.Concatenate()([x, x_23])

    x = DarknetConv2D_BN_Leaky(x,256, (3, 3))
    output_1 = DarknetConv2D(x,len(mask[1]) * (num_classes + 5), (1, 1),name = "out1")
    
    model = tf.keras.Model(inputs, [output_0, output_1], name=name)


    return model








from absl import app, flags
from core.utils import decode_cfg, load_weights
from core.image import draw_bboxes, preprocess_image, postprocess_image, read_image, read_video, Shader

import time
import cv2
import numpy as np
import tensorflow as tf

cfg = decode_cfg("cfgs/coco_yolov4_tiny.yaml")


iou_threshold = cfg["yolo"]["iou_threshold"]
score_threshold = cfg["yolo"]["score_threshold"]
max_outputs = cfg["yolo"]["max_boxes"]
num_classes = cfg["yolo"]["num_classes"]
strides = cfg["yolo"]["strides"]
mask = cfg["yolo"]["mask"]
anchors = cfg["yolo"]["anchors"]




model = YOLOv4_Tiny(cfg,416)

model.summary()

carp = cv2.imread("car.jpg")
carp = cv2.resize(carp,(416,416))
carp = (carp.astype(np.float32))/255
car = cv2.cvtColor(carp,cv2.COLOR_BGR2RGB)
car = np.expand_dims(car,0)

car.shape

pred = model.predict(car)

# pred.shape

# pred = model(tf.keras.layers.Input((416,416,3)))
outputs = Header(num_classes = num_classes, anchors = anchors, mask = mask, strides = strides, max_outputs= max_outputs, iou_threshold = iou_threshold, score_threshold = score_threshold,inputs = pred )











model.compile(optimizer="Adam", loss='categorical_crossentropy', metrics=['accuracy'])

# Saving your model to disk allows you to use it later
model.save('yolov4.h5')


# Import keras2onnx and onnx
import onnx
import keras2onnx

# Load the keras model
model = tf.keras.models.load_model('yolov4.h5')

onnx_model = keras2onnx.convert_keras(model, model.name)

onnx.save_model(onnx_model, 'yolov4.onnx')

model = onnx.load('yolov4.onnx')
print(model)


import cv2
import numpy as np

cv2.__version__


net = cv2.dnn.readNetFromONNX('yolov4.onnx')


img = cv2.imread("/home/crazy/Downloads/panda3.jpg")

img = cv2.resize(img,(32,32))
 
# Convert BGR TO RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
# Normalize the image and format it
img = np.array([img]).astype('float64') / 255.0
net.setInput(img)

Out = net.forward(["out0","out1"])

Out[1].shape













tf.keras.utils.plot_model(model,"yolov4.png")


init_weight_path = cfg['test']['init_weight_path']
if init_weight_path:
    print('Load Weights File From:', init_weight_path)
    load_weights(model, init_weight_path)
else:
    raise SystemExit('init_weight_path is Empty !')

# assign colors for difference labels
shader = Shader(cfg['yolo']['num_classes'])
names = cfg['yolo']['names']
image_size = cfg['test']['image_size'][0]





from tensorflow.keras.layers import *

def mymodel():
    x = Input((32,32,3))
    x1 = Conv2D(filters = 32,kernel_size = 3,strides = 2)(x)
    x2 = Conv2D(filters = 32,kernel_size = 3,strides = 1)(x)
    return tf.keras.models.Model(x,[x1,x2])

model  = mymodel()
    

model.summary()













import tensorflow as tf
print(tf.__version__)

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools import optimize_for_inference_lib

full_model = tf.function(lambda x: model(x))

full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="yourInputName"))

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]


for layer in layers:
    print(layer)

print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="",
                  name="smallmodel.pb",
                  as_text=False)

tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="",
                  name="smallmodel.pbtxt",
                  as_text=True)
import cv2


tensorflowNet = cv2.dnn.readNetFromTensorflow("smallmodel.pb")


# cv2.dnn.writeTextGraph("frozen_graph_yolov4try.pb", 'graph.pbtxt')

# tensorflowNet = cv2.dnn.readNetFromTensorflow("frozen_graph_yolov4try.pb")




img = cv2.imread("/home/crazy/Downloads/panda3.jpg")

blob = cv2.dnn.blobFromImage(img, size=(32, 32), swapRB=True, crop=False)

blob.shape


tensorflowNet.setInput(blob)

# Runs a forward pass to compute the net output
networkOutput = tensorflowNet.forward()

networkOutput.shape

for output in networkOutput:
    print(output.shape)

pred[1].shape

networkOutput.shape


outputs = Header(num_classes = num_classes, anchors = anchors, mask = mask, 
                 strides = strides, max_outputs= max_outputs, iou_threshold = iou_threshold,
                 score_threshold = score_threshold,inputs = networkOutput)


# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     ms, bboxes, scores, classes = inference(frame)
#     image = draw_bboxes(frame, bboxes, scores, classes, names, shader)
    
#     print('Inference Time:', ms, 'ms')
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
