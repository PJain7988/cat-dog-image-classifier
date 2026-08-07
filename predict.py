import os
import numpy as np
from PIL import Image
import onnxruntime as ort

def load_trained_model(model_path='models/cat_dog_classifier.onnx'):
    """Loads the trained ONNX model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}. Please run convert.py first.")
    
    session = ort.InferenceSession(model_path)
    return session

def predict_image(session, img_path, target_size=(150, 150)):
    """
    Predicts whether a single image is a cat or a dog using ONNX Runtime.
    Returns the predicted class and confidence.
    """
    # Load and preprocess image using PIL and Numpy
    img = Image.open(img_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    img_array = img_array / 255.0 # Rescale like in training
    
    # Get input name for the ONNX session
    input_name = session.get_inputs()[0].name
    
    # Invoke the model
    predictions = session.run(None, {input_name: img_array})
    
    # Get the prediction value
    prediction = predictions[0][0][0]
    
    if prediction > 0.5:
        return "Dog", float(prediction * 100)
    else:
        return "Cat", float((1 - prediction) * 100)

def predict_batch(session, directory_path, target_size=(150, 150)):
    """Predicts a batch of images from a directory."""
    results = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory_path, filename)
            pred_class, conf = predict_image(session, img_path, target_size)
            results[filename] = {"class": pred_class, "confidence": conf}
    return results

if __name__ == "__main__":
    # Example usage for testing
    try:
        session = load_trained_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(e)
