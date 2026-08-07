import os
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

def load_trained_model(model_path='models/cat_dog_classifier.tflite'):
    """Loads the trained TFLite model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found at {model_path}. Please run convert.py first.")
    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_image(interpreter, img_path, target_size=(150, 150)):
    """
    Predicts whether a single image is a cat or a dog using TFLite.
    Returns the predicted class and confidence.
    """
    # Load and preprocess image using PIL and Numpy
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    img_array = img_array / 255.0 # Rescale like in training
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Invoke
    interpreter.invoke()
    
    # Get the prediction
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    if prediction > 0.5:
        return "Dog", prediction * 100
    else:
        return "Cat", (1 - prediction) * 100

def predict_batch(interpreter, directory_path, target_size=(150, 150)):
    """Predicts a batch of images from a directory."""
    results = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory_path, filename)
            pred_class, conf = predict_image(interpreter, img_path, target_size)
            results[filename] = {"class": pred_class, "confidence": conf}
    return results

if __name__ == "__main__":
    # Example usage for testing
    try:
        model = load_trained_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(e)
