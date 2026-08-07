# Cat vs Dog Image Classifier

This project is a complete Deep Learning pipeline for classifying images of cats and dogs. It was built from scratch using TensorFlow and Keras, implementing a Convolutional Neural Network (CNN).

## Project Features
- **CNN Architecture from Scratch**: Developed a custom multi-layer CNN with Max Pooling and Dropout.
- **Data Augmentation**: Integrated dynamic data augmentation to prevent overfitting.
- **Model Evaluation**: Generates accuracy/loss graphs, classification reports, and a confusion matrix.
- **Interactive UI**: A simple Streamlit web application to upload custom images and get real-time predictions with confidence scores.

## Project Structure
- `model.py`: Defines the CNN architecture.
- `train.py`: Handles data downloading, preprocessing, training, evaluation, and saving the model/metrics.
- `predict.py`: Utility functions for loading the model and making predictions.
- `app.py`: The Streamlit User Interface.
- `requirements.txt`: Python dependencies.

## How to Run

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
The model must be trained first. This script will download the dataset and begin training. Depending on your hardware, this may take several minutes.
```bash
python train.py
```
*Note: The trained model will be saved to `models/cat_dog_classifier.keras` and evaluation graphs will be saved to the `static/` folder.*

### 3. Launch the Web Interface
Once the model is trained, launch the Streamlit app to test it interactively!
```bash
streamlit run app.py
```
