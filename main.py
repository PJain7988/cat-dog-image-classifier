import streamlit as st
import os
from PIL import Image
import numpy as np

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from predict import load_trained_model, predict_image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1.5rem;
    }
    
    /* Prediction Box Styling */
    .prediction-box-cat {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        color: #333;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    
    .prediction-box-dog {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        color: #333;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    
    /* Confidence Text */
    .confidence-text {
        font-size: 16px;
        font-weight: normal;
        margin-top: 10px;
        color: #555;
    }
    
    /* Hide Streamlit Main Menu & Footer if desired */
    /* #MainMenu {visibility: hidden;} */
    /* footer {visibility: hidden;} */
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR FOR METRICS ---
st.sidebar.header("📊 Model Metrics")
history_path = "static/training_history.png"
cm_path = "static/confusion_matrix.png"

if os.path.exists(history_path) and os.path.exists(cm_path):
    st.sidebar.image(history_path, caption="Training Accuracy & Loss", use_container_width=True)
    st.sidebar.image(cm_path, caption="Confusion Matrix", use_container_width=True)
else:
    st.sidebar.info("Model training metrics will appear here after you run `train.py` in your terminal.")

# --- APP HEADER ---
st.title("🐾 Cat vs Dog Image Classifier")
st.markdown("<p style='text-align: center; color: #666; font-size: 18px;'>Upload an image, and our AI will tell you if it's a Cat or a Dog!</p>", unsafe_allow_html=True)
st.markdown("---")

# --- LOAD MODEL ---
@st.cache_resource(show_spinner="Loading model into memory...")
def get_model():
    try:
        model = load_trained_model()
        return model, True, ""
    except Exception as e:
        return None, False, str(e)

model, model_loaded, error_msg = get_model()

if not model_loaded:
    st.error(f"⚠️ Error loading model! Details: {error_msg}")
    st.info("Please make sure you have run `train.py` to generate `models/cat_dog_classifier.keras`.")
    st.stop()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Add a predict button for better UX
    if st.button("Predict 🚀", use_container_width=True, type="primary"):
        with st.spinner("Analyzing image..."):
            try:
                # Save uploaded file temporarily for prediction
                temp_path = "temp_upload.jpg"
                image.convert('RGB').save(temp_path)
                
                # Make prediction
                pred_class, confidence = predict_image(model, temp_path)
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                # Display Results beautifully
                if pred_class == "Cat":
                    st.markdown(f"""
                    <div class="prediction-box-cat">
                        🐱 It's a {pred_class}!
                        <div class="confidence-text">Confidence: {confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box-dog">
                        🐶 It's a {pred_class}!
                        <div class="confidence-text">Confidence: {confidence:.2f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Progress bar for visual feedback
                st.progress(int(confidence) / 100)
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
