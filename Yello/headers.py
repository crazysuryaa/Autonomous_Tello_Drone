#header to convert outputs of model into boxes, scores, classes, valid 

import tensorflow as tf
import numpy as np

def YoloV4Header(num_classes, anchorlist, mask, strides,
                 max_outputs, iou_threshold, score_threshold,inputs):
    boxes, objects, classes = [], [], []
    dtype = inputs[0].dtype

    
    for i, logits in enumerate(inputs):
        print(i,mask[i])
        stride = strides[i]
        anchors = anchorlist[mask[i]]
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