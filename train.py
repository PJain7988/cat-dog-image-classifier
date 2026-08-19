import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def plot_training_history(history, save_path='static/training_history.png'):
    """Plots and saves the training accuracy and loss."""
    os.makedirs('static', exist_ok=True)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")

def train():
    # 1. Download and Extract Data
    print("Downloading and preparing dataset...")
    
    # Fix HTTP 403 Forbidden error by adding a User-Agent
    import urllib.request
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')]
    urllib.request.install_opener(opener)
    
    URL = 'https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip'
    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=URL, extract=True)
    
    # Handle keras extraction path which can sometimes append '_extracted'
    PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs')
    extracted_path = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_extracted', 'cats_and_dogs')
    if os.path.exists(extracted_path):
        PATH = extracted_path

    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')
    test_dir = os.path.join(PATH, 'test')

    batch_size = 32 # Adjusted batch size for smoother training
    epochs = 15
    IMG_HEIGHT = 150
    IMG_WIDTH = 150

    # 2. Data Generators and Augmentation
    print("Configuring Data Generators...")
    train_image_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_image_generator = ImageDataGenerator(rescale=1./255)
    test_image_generator = ImageDataGenerator(rescale=1./255)

    train_data_gen = train_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=train_dir,
        shuffle=True,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='binary'
    )

    val_data_gen = validation_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='binary'
    )
    
    test_data_gen = test_image_generator.flow_from_directory(
        batch_size=batch_size,
        directory=PATH,
        classes=['test'],
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode=None,
        shuffle=False
    )

    # 3. Model Creation
    print("Creating CNN Model...")
    model = create_model((IMG_HEIGHT, IMG_WIDTH, 3))
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 4. Training
    print("Starting Training...")
    history = model.fit(
        train_data_gen,
        epochs=epochs,
        validation_data=val_data_gen
    )

    # 5. Visualizing and Saving
    plot_training_history(history)
    
    os.makedirs('models', exist_ok=True)
    model.save('models/cat_dog_classifier.keras')
    print("Model saved to models/cat_dog_classifier.keras")

    # 6. Evaluation on Validation Set
    print("\\n--- Model Evaluation ---")
    val_loss, val_acc = model.evaluate(val_data_gen, verbose=0)
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")

    # Evaluate predictions for confusion matrix
    # We reset the generator to ensure it starts from the beginning
    val_data_gen.reset() 
    predictions = model.predict(val_data_gen)
    y_pred = [1 if p > 0.5 else 0 for p in predictions]
    y_true = val_data_gen.classes

    print("\\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Cat', 'Dog']))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs('static', exist_ok=True)
    plt.savefig('static/confusion_matrix.png')
    print("Confusion matrix saved to static/confusion_matrix.png")

if __name__ == '__main__':
    train()
