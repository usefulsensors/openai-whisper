import tensorflow as tf
import numpy as np
tflite_model_path_1 = '/home/niranjanyadla/Downloads/whisper-encoder.tflite'


# Load the TFLite model and allocate tensors
interpreter_1 = tf.lite.Interpreter(model_path=tflite_model_path_1)
interpreter_1.allocate_tensors()

print("== Input details ==")
print("name:", interpreter_1.get_input_details()[0]['name'])
print("shape:", interpreter_1.get_input_details()[0]['shape'])
print("type:", interpreter_1.get_input_details()[0]['dtype'])

print("\nDUMP INPUT")
print(interpreter_1.get_input_details()[0])

print("\n== Output details ==")
print("name:", interpreter_1.get_output_details()[0]['name'])
print("shape:", interpreter_1.get_output_details()[0]['shape'])
print("type:", interpreter_1.get_output_details()[0]['dtype'])

print("\nDUMP OUTPUT")
print(interpreter_1.get_output_details()[0])

# Get input and output tensors
input_details = interpreter_1.get_input_details()
output_details = interpreter_1.get_output_details()

# Test the model with random data
input_shape = input_details[0]['shape']

print("before whisper-encoder.tflite inference")
interpreter_1.invoke()
print("after whisper-encoder.tflite inference")
print("\n\n\n\n")

tflite_model_path_2 = '/home/niranjanyadla/Downloads/whisper-decoder_main.tflite'
# Load the TFLite model and allocate tensors
interpreter_2 = tf.lite.Interpreter(model_path=tflite_model_path_2)
interpreter_2.allocate_tensors()

print("== Input details ==")
print("name:", interpreter_2.get_input_details()[0]['name'])
print("shape:", interpreter_2.get_input_details()[0]['shape'])
print("type:", interpreter_2.get_input_details()[0]['dtype'])

print("\nDUMP INPUT")
print(interpreter_2.get_input_details()[0])

print("\n== Output details ==")
print("name:", interpreter_2.get_output_details()[0]['name'])
print("shape:", interpreter_2.get_output_details()[0]['shape'])
print("type:", interpreter_2.get_output_details()[0]['dtype'])

print("\nDUMP OUTPUT")
print(interpreter_2.get_output_details()[0])

# Get input and output tensors
input_details = interpreter_2.get_input_details()
output_details = interpreter_2.get_output_details()

# Test the model with random data
input_shape = input_details[0]['shape']

print("before whisper-decoder_main.tflite inference")
interpreter_2.invoke()
print("after whisper-decoder_main.tflite inference")
print("\n\n\n\n")

tflite_model_path_3 = '/home/niranjanyadla/Downloads/whisper-decoder_language.tflite'
# Load the TFLite model and allocate tensors
interpreter_3 = tf.lite.Interpreter(model_path=tflite_model_path_3)
interpreter_3.allocate_tensors()

print("== Input details ==")
print("name:", interpreter_3.get_input_details()[0]['name'])
print("shape:", interpreter_3.get_input_details()[0]['shape'])
print("type:", interpreter_3.get_input_details()[0]['dtype'])

print("\nDUMP INPUT")
print(interpreter_3.get_input_details()[0])

print("\n== Output details ==")
print("name:", interpreter_3.get_output_details()[0]['name'])
print("shape:", interpreter_3.get_output_details()[0]['shape'])
print("type:", interpreter_3.get_output_details()[0]['dtype'])

print("\nDUMP OUTPUT")
print(interpreter_3.get_output_details()[0])

# Get input and output tensors
input_details = interpreter_3.get_input_details()
output_details = interpreter_3.get_output_details()

# Test the model with random data
input_shape = input_details[0]['shape']

print("before whisper-decoder_language.tflite inference")
interpreter_3.invoke()
print("after whisper-decoder_language.tflite inference")
