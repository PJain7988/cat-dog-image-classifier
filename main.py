import streamlit as st
import os
from PIL import Image
from predict import load_trained_model, predict_image

# --- Page Config ---
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide"
)

# --- Title and Header ---
st.title("🐾 Cat vs Dog Image Classifier")
st.markdown("Upload an image of a cat or a dog, and our Convolutional Neural Network will predict what it is!")

# --- Sidebar for Metrics ---
st.sidebar.header("Model Evaluation Metrics")
st.sidebar.markdown("Once you train the model using `train.py`, the training metrics will appear below.")

# Display training history if available
history_path = "static/training_history.png"
if os.path.exists(history_path):
    st.sidebar.image(history_path, caption="Training Accuracy & Loss", use_container_width=True)
else:
    st.sidebar.warning("Training history not found. Run `python train.py` to generate.")

# Display confusion matrix if available
cm_path = "static/confusion_matrix.png"
if os.path.exists(cm_path):
    st.sidebar.image(cm_path, caption="Confusion Matrix", use_container_width=True)

# --- Main App Logic ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        
        # Save temporary image for prediction
        temp_img_path = "temp_upload.jpg"
        image.save(temp_img_path)

with col2:
    st.header("2. Prediction")
    if uploaded_file is not None:
        if st.button("Predict 🚀"):
            with st.spinner("Analyzing image..."):
                try:
                    # Load model
                    model = load_trained_model()
                    
                    # Predict
                    pred_class, confidence = predict_image(model, temp_img_path)
                    
                    # Display results
                    st.success("Prediction Complete!")
                    st.markdown(f"### **Prediction:** {pred_class}")
                    st.markdown(f"### **Confidence:** {confidence:.2f}%")
                    
                    # Progress bar for visual confidence
                    st.progress(int(confidence))
                    
                except FileNotFoundError:
                    st.error("Model not found! Please run `python train.py` first to train and save the model.")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists("temp_upload.jpg"):
                        os.remove("temp_upload.jpg")
    else:
        st.info("Upload an image on the left to see the prediction here.")
