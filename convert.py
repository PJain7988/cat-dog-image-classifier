import tensorflow as tf
import os

def convert_model():
    keras_model_path = 'models/cat_dog_classifier.keras'
    tflite_model_path = 'models/cat_dog_classifier.tflite'
    
    if not os.path.exists(keras_model_path):
        print(f"Error: {keras_model_path} does not exist.")
        print("Please run `python train.py` first to train the model!")
        return

    print("Loading trained Keras model...")
    model = tf.keras.models.load_model(keras_model_path)
    
    print("Converting to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional: Quantization to make it even smaller
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Successfully converted and saved TFLite model to {tflite_model_path}")
    print(f"Size: {os.path.getsize(tflite_model_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    convert_model()
