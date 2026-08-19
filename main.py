import streamlit as st
import os
from PIL import Image
import numpy as np

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from predict import load_trained_model, predict_image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise Vision AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CORPORATE SAAS CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global styling */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #f4f7f6;
        color: #2c3e50;
    }
    
    /* Header and Subheader */
    h1 {
        color: #1a252f;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        display: inline-block;
    }
    
    .subtitle {
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-top: 10px;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* Upload Box */
    div[data-testid="stFileUploader"] {
        border: 1.5px dashed #bdc3c7;
        border-radius: 8px;
        padding: 20px;
        background-color: #ffffff;
        transition: border-color 0.3s ease;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #3498db;
    }
    
    /* Structured Data Cards for Prediction */
    .data-card {
        background: #ffffff;
        border: 1px solid #e0e6ed;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        margin-top: 20px;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    
    .card-header {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #7f8c8d;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    .pred-result {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 5px;
    }
    
    .confidence {
        font-size: 1.1rem;
        color: #34495e;
        font-weight: 500;
        margin-top: 10px;
    }
    
    /* Status indicators */
    .status-badge-cat {
        background-color: #e8f4fd;
        color: #2980b9;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }

    .status-badge-dog {
        background-color: #fdf2e9;
        color: #d35400;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    
    /* Button Styling */
    button[kind="primary"] {
        background-color: #2980b9 !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        transition: background-color 0.2s ease !important;
        width: 100%;
    }
    
    button[kind="primary"]:hover {
        background-color: #3498db !important;
    }

    /* Metric Containers */
    .metric-container {
        display: flex;
        justify-content: space-between;
        width: 100%;
        border-top: 1px solid #ecf0f1;
        padding-top: 15px;
        margin-top: 15px;
    }
    
    .metric-label {
        color: #95a5a6;
        font-size: 0.9rem;
    }
    
    .metric-value {
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR FOR METRICS ---
with st.sidebar:
    st.markdown("<h3 style='color: #2c3e50; font-weight: 600;'>System Diagnostics</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    history_path = "static/training_history.png"
    cm_path = "static/confusion_matrix.png"

    if os.path.exists(history_path) and os.path.exists(cm_path):
        st.markdown("**Model Performance**")
        st.image(history_path, use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Confusion Matrix Analysis**")
        st.image(cm_path, use_container_width=True)
    else:
        st.info("Performance telemetry is currently unavailable. Run `train.py` to generate system metrics.")

# --- APP HEADER ---
st.markdown("<h1>Enterprise Vision AI Platform</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Automated image classification system for binary entity resolution (Felis catus vs. Canis lupus familiaris).</div>", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource(show_spinner="Loading inference engine...")
def get_model():
    try:
        model = load_trained_model()
        return model, True, ""
    except Exception as e:
        return None, False, str(e)

model, model_loaded, error_msg = get_model()

if not model_loaded:
    st.error(f"System Error: Inference engine failed to initialize. Details: {error_msg}")
    st.info("Action Required: Execute `train.py` to compile the classification model.")
    st.stop()

# --- MAIN LAYOUT ---
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # --- FILE UPLOADER ---
    uploaded_file = st.file_uploader("Upload visual payload for processing", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image inside a card
        try:
            image = Image.open(uploaded_file)
            st.markdown("<div class='data-card' style='padding: 15px;'>", unsafe_allow_html=True)
            st.image(image, use_container_width=True, output_format="PNG")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Data ingestion failed: {e}")
            st.stop()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Add a predict button
        if st.button("Execute Image Classification", type="primary"):
            with st.spinner("Processing data through inference engine..."):
                try:
                    # Save uploaded file temporarily for prediction
                    temp_path = "temp_upload.jpg"
                    image.convert('RGB').save(temp_path)
                    
                    # Make prediction
                    pred_class, confidence = predict_image(model, temp_path)
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    # Premium Corporate Display Results
                    badge_class = "status-badge-cat" if pred_class == "Cat" else "status-badge-dog"
                    
                    st.markdown(f"""
                    <div class="data-card">
                        <div class="card-header">Classification Output</div>
                        <div class="{badge_class}">Status: Identified</div>
                        <div class="pred-result">{pred_class}</div>
                        
                        <div class="metric-container">
                            <span class="metric-label">Confidence Score</span>
                            <span class="metric-value">{confidence:.2f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Minimal corporate progress bar
                    st.progress(int(confidence) / 100)
                    
                except Exception as e:
                    st.error(f"Inference execution failed: {str(e)}")
