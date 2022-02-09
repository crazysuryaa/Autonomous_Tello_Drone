
from core.utils import decode_cfg, load_weights
from core.image import draw_bboxes, preprocess_image, postprocess_image, read_image, read_video, Shader
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
import tensorflow as tf
import sys


#header file to convert model output to into boxes, scores, classes, valid 
from headers import YoloV4Header as Header
from core.model.one_stage.yolov4 import YOLOv4_Tiny as Model
cfg = decode_cfg("cfgs/coco_yolov4_tiny.yaml")
model,evalmodel = Model(cfg,416)





#importing model yolov4
# from core.model.one_stage.yolov4 import YOLOv4 as Model
# cfg = decode_cfg("cfgs/coco_yolov4.yaml")
# model,evalmodel = Model(cfg,416)



#loading weights from path

init_weight_path = cfg['test']['init_weight_path']
if init_weight_path:
    print('Load Weights File From:', init_weight_path)
    load_weights(model, init_weight_path)
else:
    raise SystemExit('init_weight_path is Empty !')


model.save("yolov4save.h5")




converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                         tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
with open('testing1.tflite', 'wb') as f:
    f.write(tflite_model)


# tflite_model = converter.convert()
# open("test.tflite", "wb").write(tflite_model)









## getting error resize nearest neighbours no idea what it is


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="testing1.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#
#
# blazeface_tf = tf.keras.models.load_model("yolov4save.h5")
#
#
#
# facemesh_tf = tf.keras.models.load_model("./kerasmodels/facemesh_tf.h5")
#
#
# faceinterpreter = tf.lite.Interpreter(model_path="./tflitemodels/newface.tflite")
# facemeshinterpreter = tf.lite.Interpreter(model_path="./tflitemodels/newfacemesh.tflite")
#
# facemeshinterpreter.allocate_tensors()
# faceinterpreter.allocate_tensors()
#
# facemeshinput_details = facemeshinterpreter.get_input_details()
# facemeshoutput_details = facemeshinterpreter.get_output_details()
#
# faceinput_details = faceinterpreter.get_input_details()
# faceoutput_details = faceinterpreter.get_output_details()
#


shader = Shader(cfg['yolo']['num_classes'])
names = cfg['yolo']['names']
image_size = cfg['test']['image_size'][0]
iou_threshold = cfg["yolo"]["iou_threshold"]
score_threshold = cfg["yolo"]["score_threshold"]
max_outputs = cfg["yolo"]["max_boxes"]
num_classes = cfg["yolo"]["num_classes"]
strides = cfg["yolo"]["strides"]
mask = cfg["yolo"]["mask"]
anchors = cfg["yolo"]["anchors"]


#freezing the model and saving .pb file

#opencvdnnmodel name 
cvname = "testingdnn"
from interference import Interferencemodels
Interferencemodels(model,name = cvname )



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




#predicting model directly using tensorflow model

output = image = cv2.imread("person1.jpeg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = preprocess_image(image, (416,416))
h, w = image.shape[:2]
plt.imshow(image)
img = np.expand_dims(image,0)
img = img/255.
pred = model.predict(img)
bboxes, scores, classes, valid_detections = Header(80, anchors, mask, strides, 100,
                 iou_threshold, score_threshold,inputs = pred)

bboxes = bboxes[0][:valid_detections[0]]
scores = scores[0][:valid_detections[0]]
classes = classes[0][:valid_detections[0]]

# bboxes *= image_size
_, bboxes = postprocess_image(image, (w, h), bboxes.numpy())

image = draw_bboxes(image, bboxes, scores, classes, names, shader)

plt.imshow(image)





#########predicting from tflite


output = image = cv2.imread("person1.jpeg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = preprocess_image(image, (416,416))
h, w = image.shape[:2]
img = np.expand_dims(image,0).astype(np.float32)
img = img/255.
#plt.imshow(image)
# output = image = cv2.imread("person1.jpeg")
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# # image = preprocess_image(image, (416,416))
# image = cv2.resize(image,(416,416))
# target_input = np.expand_dims(image.copy(), axis=0).astype(np.float32)
img= np.ascontiguousarray(img)
plt.imshow(img[0])
# image = target_input
#
# h, w = image.shape[:2]
# plt.imshow(image[0])
# img = np.expand_dims(image,0)
#



pred[0].shape

interpreter = tf.lite.Interpreter(model_path="testing1.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], img)

interpreter.invoke()

pred = [interpreter.get_tensor(output_details[0]['index']),
        interpreter.get_tensor(output_details[1]['index'])]



bboxes, scores, classes, valid_detections = Header(80, anchors, mask, strides, 100,
                 iou_threshold, score_threshold,inputs = pred)

bboxes = bboxes[0][:valid_detections[0]]
scores = scores[0][:valid_detections[0]]
classes = classes[0][:valid_detections[0]]

# bboxes *= image_size
_, bboxes = postprocess_image(image, (w, h), bboxes.numpy())

image = draw_bboxes(image, bboxes, scores, classes, names, shader)

plt.imshow(image)







#reading and predicting model directly using opencvdnn 
#This is where code is not working!!!!

name = cvname
tensorflowNet = cv2.dnn.readNetFromTensorflow(name+"/"+name +".pb")
outNames = tensorflowNet.getUnconnectedOutLayersNames()



output = image = cv2.imread("person1.jpeg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = preprocess_image(image, (416,416))
h, w = image.shape[:2]
img = np.expand_dims(image,0)
img = img/255.

#reshape because opencv dnn takes input as (1,channels , w,h) instead of (1,w,h,channels)

imgcv = np.reshape(img,(1,3,416,416),order= 'C')

# imgcv = cv2.dnn.blobFromImage(image, size=(416, 416),scalefactor=1.0/255)

tensorflowNet.setInput(imgcv)
predcv = tensorflowNet.forward([outNames[1],outNames[0]])
predcv[0].shape

#reshape because opencv dnn gives output as (1,channels , w,h) instead of (1,w,h,channels)

predcv[0] = np.reshape(predcv[0],(1,13,13,255),order='C')
predcv[1] = np.reshape(predcv[1],(1,26,26,255),order='C')

bboxes, scores, classes, valid_detections = Header(80, anchors, mask, strides, 100,
                 iou_threshold, score_threshold,inputs = predcv)

bboxes = bboxes[0][:valid_detections[0]]
scores = scores[0][:valid_detections[0]]
classes = classes[0][:valid_detections[0]]

# bboxes *= image_size
_, bboxes = postprocess_image(image, (w, h), bboxes.numpy())

image = draw_bboxes(image, bboxes, scores, classes, names, shader)

plt.imshow(image)














######for tflite converted model This is not working at all
#Getting error while converting itself



##saving into tflite model

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("test.tflite", "wb").write(tflite_model)

## getting error resize nearest neighbours no idea what it is


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="test.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[1]['index'])
print(output_data)









################################################################Running on real time video#####################









def cvinference(image):
    h, w = image.shape[:2]
    image = preprocess_image(image, (image_size, image_size)).astype(np.float32)
    images = np.expand_dims(image, axis=0)

    tic = time.time()
    bboxes, scores, classes, valid_detections = Header(80, anchors, mask, strides, 10,
                 iou_threshold, score_threshold,inputs = pred)
    
    toc = time.time()

    bboxes = bboxes[0][:valid_detections[0]]
    scores = scores[0][:valid_detections[0]]
    classes = classes[0][:valid_detections[0]]

    # bboxes *= image_size
    _, bboxes = postprocess_image(image, (w, h), bboxes.numpy())

    return (toc - tic) * 1000, bboxes, scores, classes






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

    # bboxes *= image_size
    _, bboxes = postprocess_image(image, (w, h), bboxes.numpy())

    return (toc - tic) * 1000, bboxes, scores, classes




interpreter = tf.lite.Interpreter(model_path="testing1.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

bboxes, scores, classes, valid_detections = Header(80, anchors, mask, strides, 100,
                 iou_threshold, score_threshold,inputs = pred)

def tfliteinference(image):
    h, w = image.shape[:2]
    image = preprocess_image(image, (image_size, image_size)).astype(np.float32)
    images = np.expand_dims(image, axis=0).astype(np.float32)
    images = images / 255.
    # img = np.ascontiguousarray(images)
    tic = time.time()
    interpreter.set_tensor(input_details[0]['index'], images)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[0]['index']),
            interpreter.get_tensor(output_details[1]['index'])]

    bboxes, scores, classes, valid_detections = Header(80, anchors, mask, strides, 10,
                                                       iou_threshold, score_threshold, inputs=pred)

    # bboxes, scores, classes, valid_detections = evalmodel.predict(images)

    toc = time.time()

    bboxes = bboxes[0][:valid_detections[0]]
    scores = scores[0][:valid_detections[0]]
    classes = classes[0][:valid_detections[0]]

    # bboxes *= image_size
    _, bboxes = postprocess_image(image, (w, h), bboxes.numpy())

    return (toc - tic) * 1000, bboxes, scores, classes


#######for direct tensorflow model



cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ms, bboxes, scores, classes = inference(frame)
    image = draw_bboxes(frame, bboxes, scores, classes, names, shader)
    
    print('Inference Time:', ms, 'ms')
    print('Fps:', 1000/ms)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()











# ###################3tflite interference
# cap = cv2.VideoCapture(0)

# while (True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     ms, bboxes, scores, classes = tfliteinference(frame)
#     image = draw_bboxes(frame, bboxes, scores, classes, names, shader)

#     print('Inference Time:', ms, 'ms')
#     print('Fps:', 1000 / ms)
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

#######for converted opencvdnn model This is not working properly



# import cv2

# from headers import YoloV4Header as Header

# from interference import Interferencemodels

# Interferencemodels(model,name = cvname )

# name = cvname

# tensorflowNet = cv2.dnn.readNetFromTensorflow(name+"/"+name +".pb")
# outNames = tensorflowNet.getUnconnectedOutLayersNames()

# cap = cv2.VideoCapture(0)
# image_size = 416
# while(True):
#     # Capture frame-by-frame
#     ret, image = cap.read()
#     # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     image = preprocess_image(image, (416,416))
#     h, w = image.shape[:2]
#     img = np.expand_dims(image,0)
#     img = img/255.
    
#     #reshape because opencv dnn takes input as (1,channels , w,h) instead of (1,w,h,channels)
    
#     imgcv = np.reshape(img,(1,3,416,416),order= 'C')
    
#     # imgcv = cv2.dnn.blobFromImage(image, size=(416, 416),scalefactor=1.0/255)
#     tic = time.time()
#     tensorflowNet.setInput(imgcv)
    
#     pred = tensorflowNet.forward([outNames[1],outNames[0]])
#     pred[0] = np.reshape(pred[0],(1,13,13,255))
#     pred[1] = np.reshape(pred[1],(1,26,26,255))
    
    
#     bboxes, scores, classes, valid_detections = Header(80, anchors, mask, strides, 100,
#                   iou_threshold, score_threshold,inputs = pred)
    
#     toc = time.time()

#     bboxes = bboxes[0][:valid_detections[0]]
#     scores = scores[0][:valid_detections[0]]
#     classes = classes[0][:valid_detections[0]]

#     # bboxes *= image_size
#     _, bboxes = postprocess_image(image, (416, 416), bboxes.numpy())
    
#     ms, bboxes, scores, classes = (toc - tic) * 1000, bboxes, scores, classes
#     image = draw_bboxes(image, bboxes, scores, classes, names, shader)
#     print('Inference Time:', ms, 'ms')
#     print('Fps:', 1000/ms)
#     cv2.imshow('frame',image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cap.release()
# cv2.destroyAllWindows()




# import numpy as np

# img = cv2.imread("home.png")
# carp = cv2.resize(img,(416,416))
# car = cv2.cvtColor(carp,cv2.COLOR_BGR2RGB)
# car = np.expand_dims(car,0)
# car = car/255
# plt.imshow(car[0])

# # blob = cv2.dnn.blobFromImage(car, size=(416, 416))
# car = np.reshape(car,(1,3,416,416))


# tensorflowNet.setInput(car)
# pred = tensorflowNet.forward([outNames[1],outNames[0]])




# import matplotlib.pyplot as plt

# plt.imshow(pred[0][0][0])




# pred[0] = np.reshape(pred[0],(1,13,13,255))
# pred[1] = np.reshape(pred[1],(1,26,26,255))
# for p in pred:
#     print(p.shape)




# bboxes, scores, classes, valid_detections = Header(80, anchors, mask, strides, 10,
#               iou_threshold, score_threshold,inputs = pred)



# predtf = model(cartf)
# bboxes, scores, classes, valid_detections = Header(80, anchors, mask, strides, 10,
#               iou_threshold, score_threshold,inputs = predtf)






# import onnx
# import keras2onnx

# # Load the keras model
# model = tf.keras.models.load_model(name )

# onnx_model = keras2onnx.convert_keras(model, model.name)

# onnx.save_model(onnx_model, 'test10.onnx')

# onnxmodel = cv2.dnn.readNetFromONNX('test10.onnx')

# outNames = onnxmodel.getUnconnectedOutLayersNames()



# from absl import app, flags
# from core.utils import decode_cfg, load_weights
# from core.image import draw_bboxes, preprocess_image, postprocess_image, read_image, read_video, Shader
# import time
# import cv2



# cfg = decode_cfg("cfgs/coco_yolov4_tiny.yaml")

# shader = Shader(cfg['yolo']['num_classes'])
# names = cfg['yolo']['names']
# image_size = cfg['test']['image_size'][0]
# iou_threshold = cfg["yolo"]["iou_threshold"]
# score_threshold = cfg["yolo"]["score_threshold"]
# max_outputs = cfg["yolo"]["max_boxes"]
# num_classes = cfg["yolo"]["num_classes"]
# strides = cfg["yolo"]["strides"]
# mask = cfg["yolo"]["mask"]
# anchors = cfg["yolo"]["anchors"]




# cap = cv2.VideoCapture(0)
# image_size = 416
# while(True):
#     # Capture frame-by-frame
#     ret, image = cap.read()
#     img = image.astype(np.float32)/255
#     # img = image/255.
#     blob = cv2.dnn.blobFromImage(img, size=(416, 416))
#     # blob = blob/255.
    
#     print(blob)
#     tensorflowNet.setInput(blob)
#     pred = tensorflowNet.forward([outNames[1],outNames[0]])
#     pred[0] = np.reshape(pred[0],(1,13,13,255))
#     pred[1] = np.reshape(pred[1],(1,26,26,255))
#     tic = time.time()
    
#     bboxes, scores, classes, valid_detections = Header(80, anchors, mask, strides, 100,
#                   iou_threshold, score_threshold,inputs = pred)
    
#     toc = time.time()

#     bboxes = bboxes[0][:valid_detections[0]]
#     scores = scores[0][:valid_detections[0]]
#     classes = classes[0][:valid_detections[0]]

#     # bboxes *= image_size
#     _, bboxes = postprocess_image(image, (416, 416), bboxes.numpy())
    
#     ms, bboxes, scores, classes = (toc - tic) * 1000, bboxes, scores, classes
#     image = draw_bboxes(image, bboxes, scores, classes, names, shader)
#     print('Inference Time:', ms, 'ms')
#     print('Fps:', 1000/ms)
#     cv2.imshow('frame',image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# cap.release()
# cv2.destroyAllWindows()














