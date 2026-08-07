import tensorflow as tf
import tf2onnx
import os

def convert_model():
    keras_model_path = 'models/cat_dog_classifier.keras'
    onnx_model_path = 'models/cat_dog_classifier.onnx'
    
    if not os.path.exists(keras_model_path):
        print(f"Error: {keras_model_path} does not exist.")
        print("Please run `python train.py` first to train the model!")
        return

    print("Loading trained Keras model...")
    model = tf.keras.models.load_model(keras_model_path)
    
    print("Converting to ONNX format...")
    # Define the input signature based on the model's input shape
    # The batch size is None, and the image size is 150x150x3
    spec = (tf.TensorSpec((None, 150, 150, 3), tf.float32, name="input"),)
    
    # Convert from Keras to ONNX
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_model_path)
    
    print(f"Successfully converted and saved ONNX model to {onnx_model_path}")
    print(f"Size: {os.path.getsize(onnx_model_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    convert_model()
