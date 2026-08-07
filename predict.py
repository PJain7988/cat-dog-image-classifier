import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

def load_trained_model(model_path='models/cat_dog_classifier.keras'):
    """Loads the trained Keras model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    return tf.keras.models.load_model(model_path)

def predict_image(model, img_path, target_size=(150, 150)):
    """
    Predicts whether a single image is a cat or a dog.
    Returns the predicted class and confidence.
    """
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    img_array = img_array / 255.0 # Rescale like in training
    
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        return "Dog", prediction * 100
    else:
        return "Cat", (1 - prediction) * 100

def predict_batch(model, directory_path, target_size=(150, 150)):
    """Predicts a batch of images from a directory."""
    results = {}
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory_path, filename)
            pred_class, conf = predict_image(model, img_path, target_size)
            results[filename] = {"class": pred_class, "confidence": conf}
    return results

if __name__ == "__main__":
    # Example usage for testing
    try:
        model = load_trained_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(e)
