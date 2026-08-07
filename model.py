import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model(input_shape=(150, 150, 3)):
    """
    Creates and returns a Convolutional Neural Network (CNN) model for Cat vs Dog classification.
    """
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Fourth Convolutional Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Flattening and Dense Layers
        Flatten(),
        Dropout(0.5), # Dropout to prevent overfitting
        Dense(512, activation='relu'),
        
        # Output layer for binary classification (Cat or Dog)
        Dense(1, activation='sigmoid')
    ])
    
    return model
