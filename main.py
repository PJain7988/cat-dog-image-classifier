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

# --- PREMIUM CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global styling */
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e1e24 0%, #151518 100%);
        color: #ffffff;
    }
    
    /* Header and Subheader */
    h1 {
        color: #f1f2f6;
        text-align: center;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0px;
        background: -webkit-linear-gradient(45deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        text-align: center;
        color: #a4b0be;
        font-size: 1.2rem;
        margin-top: 5px;
        margin-bottom: 40px;
        font-weight: 300;
    }
    
    /* Upload Box */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #ff9a9e;
        border-radius: 15px;
        padding: 20px;
        background-color: rgba(255,255,255,0.05);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #fecfef;
        background-color: rgba(255,255,255,0.08);
    }
    
    /* Glassmorphism Prediction Box */
    .glass-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-top: 30px;
        animation: fadeIn 0.8s ease;
    }
    
    .pred-result {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 10px;
    }
    
    .cat-text {
        background: -webkit-linear-gradient(45deg, #a8edea 0%, #fed6e3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .dog-text {
        background: -webkit-linear-gradient(45deg, #f6d365 0%, #fda085 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .confidence {
        font-size: 1.2rem;
        color: #ced6e0;
        font-weight: 400;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Button Styling */
    button[kind="primary"] {
        background: linear-gradient(45deg, #ff9a9e 0%, #fecfef 100%) !important;
        border: none !important;
        color: #1e1e24 !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        transition: transform 0.2s ease !important;
    }
    
    button[kind="primary"]:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 4px 15px rgba(255, 154, 158, 0.4) !important;
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR FOR METRICS ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #f1f2f6;'>📊 Analytics</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    history_path = "static/training_history.png"
    cm_path = "static/confusion_matrix.png"

    if os.path.exists(history_path) and os.path.exists(cm_path):
        st.image(history_path, caption="Training Accuracy & Loss", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
    else:
        st.info("Model training metrics will appear here after you run `train.py`.")

# --- APP HEADER ---
st.markdown("<h1>🐾 Neural Vision</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced Cat vs Dog Image Classification using Deep Learning</div>", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource(show_spinner="Initializing Neural Network...")
def get_model():
    try:
        model = load_trained_model()
        return model, True, ""
    except Exception as e:
        return None, False, str(e)

model, model_loaded, error_msg = get_model()

if not model_loaded:
    st.error(f"⚠️ **Error initializing model:** {error_msg}")
    st.info("Ensure you have run `train.py` to generate `models/cat_dog_classifier.keras`.")
    st.stop()

# --- MAIN LAYOUT ---
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # --- FILE UPLOADER ---
    uploaded_file = st.file_uploader("Drop an image here to analyze", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        try:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Target Image", output_format="PNG")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            st.stop()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Add a predict button
        if st.button("Initialize Analysis 🚀", use_container_width=True, type="primary"):
            with st.spinner("Processing visual data..."):
                try:
                    # Save uploaded file temporarily for prediction
                    temp_path = "temp_upload.jpg"
                    image.convert('RGB').save(temp_path)
                    
                    # Make prediction
                    pred_class, confidence = predict_image(model, temp_path)
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    # Premium Display Results
                    css_class = "cat-text" if pred_class == "Cat" else "dog-text"
                    icon = "🐱" if pred_class == "Cat" else "🐶"
                    
                    st.markdown(f"""
                    <div class="glass-box">
                        <div class="pred-result {css_class}">
                            {icon} {pred_class}
                        </div>
                        <div class="confidence">
                            Confidence Level: <b>{confidence:.2f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Custom progress bar
                    st.progress(int(confidence) / 100)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
