
#interference script i made

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.tools import optimize_for_inference_lib
import cv2
import os 
import tensorflow as tf
import  tf2onnx
import shutil
def Interferencemodels(model,name = "test1" ):
    full_model = tf.function(lambda x: model(x))
    
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="yourInputName"))
    
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    
    
    for layer in layers:
        print(layer)
    # print(layers[0],layers[-1])

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    if os.path.isdir(name):
        print("removed directory")
        shutil.rmtree(name)
    os.mkdir(name)
    # os.makedirs(name, exist_ok=True)
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="",
                      name= name+"/"+name +".pb",
                      as_text=False)
    
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="",
                      name= name+"/"+name +".pbtxt",
                      as_text=True)

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file= name+"/"+name +".pb",
        input_arrays=[layers[0]],
        output_arrays=[layers[-1]]
    )
    tflite_model = converter.convert()
    open("mobilenet.tflite", "wb").write(tflite_model)



    