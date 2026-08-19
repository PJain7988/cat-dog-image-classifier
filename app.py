import streamlit as st
import os
import base64
from PIL import Image
import numpy as np

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from predict import load_trained_model, predict_image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Cat vs Dog Image Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Dark Theme Settings */
    [data-testid="stAppViewContainer"] {
        background-color: #0b0f19;
        background-image: 
            radial-gradient(at 0% 0%, rgba(76, 29, 149, 0.2) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(29, 78, 216, 0.15) 0px, transparent 50%);
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background-color: rgba(11, 15, 25, 0.8) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stApp > header {
        background-color: transparent !important;
    }

    /* Hide Streamlit branding and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Navigation Bar */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 3rem;
    }
    .nav-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 20px;
        font-weight: 700;
        color: #fff;
    }
    .nav-logo span.cat { color: #a855f7; }
    .nav-logo span.dog { color: #3b82f6; }
    
    .nav-links {
        display: flex;
        gap: 2rem;
    }
    .nav-links a {
        color: #94a3b8;
        text-decoration: none;
        font-size: 14px;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    .nav-links a:hover, .nav-links a.active {
        color: #fff;
    }
    .nav-links a.active {
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    .github-btn {
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 5px 15px;
        color: white !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Hero Section */
    .hero-title {
        text-align: center;
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 1rem;
        color: #f8fafc;
        line-height: 1.2;
    }
    .hero-title span.cat { color: #a855f7; }
    .hero-title span.dog { color: #3b82f6; }
    .hero-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }
    .hero-subtitle span.cat { color: #a855f7; font-weight: 600;}
    .hero-subtitle span.dog { color: #3b82f6; font-weight: 600;}

    /* Card Containers */
    .upload-card, .result-card {
        background-color: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        height: 100%;
        backdrop-filter: blur(12px);
    }
    
    .upload-inner {
        border: 2px dashed rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    .upload-inner:hover {
        border-color: rgba(59, 130, 246, 0.5);
    }
    
    /* Hide default file uploader text but keep functionality */
    [data-testid="stFileUploader"] section {
        padding: 0;
        background-color: transparent;
    }
    [data-testid="stFileUploader"] section > div > span {
        display: none;
    }
    
    /* Styling Streamlit button */
    .stButton > button {
        background: linear-gradient(135deg, #a855f7, #3b82f6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.5);
        color: white;
    }

    /* Result styling */
    .result-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 5px 15px;
        border-radius: 20px;
        background-color: rgba(16, 185, 129, 0.1);
        color: #10b981;
        font-size: 14px;
        font-weight: 600;
        margin-top: 15px;
    }
    
    .confidence-score {
        font-size: 2.5rem;
        font-weight: 800;
        color: #3b82f6;
        margin-top: 10px;
    }

    /* Feature Cards */
    .feature-card {
        background-color: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        display: flex;
        gap: 15px;
        align-items: flex-start;
        margin-top: 2rem;
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.1);
    }
    .feature-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        flex-shrink: 0;
    }
    .bg-purple { background-color: rgba(168, 85, 247, 0.1); color: #a855f7; }
    .bg-blue { background-color: rgba(59, 130, 246, 0.1); color: #3b82f6; }
    .bg-green { background-color: rgba(16, 185, 129, 0.1); color: #10b981; }
    .bg-pink { background-color: rgba(236, 72, 153, 0.1); color: #ec4899; }
    
    .feature-text h4 { margin: 0 0 5px 0; font-size: 16px; color: #f8fafc; }
    .feature-text p { margin: 0; font-size: 13px; color: #94a3b8; line-height: 1.5; }

    /* Footer */
    .custom-footer {
        text-align: center;
        margin-top: 4rem;
        padding-bottom: 2rem;
        color: #94a3b8;
        font-size: 14px;
    }
    .custom-footer span { color: #a855f7; font-weight: 600; }
    
    /* Progress bar style override */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        border-radius: 10px;
    }
    
    /* General Image border radius */
    img {
        border-radius: 16px;
    }

</style>
""", unsafe_allow_html=True)

# --- NAVIGATION BAR ---
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">
        🐾 <span class="cat">Cat</span> vs <span class="dog">Dog</span> <span style="font-weight: 400; font-size: 14px; color: #94a3b8; margin-left: 5px;">Image Classifier</span>
    </div>
    <div class="nav-links">
        <a href="#" class="active">Home</a>
        <a href="#">How It Works</a>
        <a href="#">About</a>
        <a href="#" class="github-btn">🐙 GitHub</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
<div class="hero-title"><span class="cat">Cat</span> vs <span class="dog">Dog</span> Image Classifier</div>
<div class="hero-subtitle">Upload an image, and our AI will tell you if it's a <span class="cat">Cat</span> or a <span class="dog">Dog</span>!</div>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def get_model():
    try:
        model = load_trained_model()
        return model, True
    except Exception as e:
        return None, False

model, model_loaded = get_model()

if not model_loaded:
    st.error("⚠️ Model not found! Please make sure you have run `train.py` to generate `models/cat_dog_classifier.keras`.")
    st.stop()

# --- MAIN CONTENT GRID ---
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="upload-card"><div class="upload-inner">', unsafe_allow_html=True)
    st.markdown("""
        <div style="font-size: 40px; margin-bottom: 10px; color: #3b82f6;">☁️</div>
        <h3 style="color: white; margin-bottom: 10px;">Upload an Image</h3>
        <p style="color: #94a3b8; font-size: 14px; margin-bottom: 20px;">Drag & drop an image here, or click to browse</p>
    """, unsafe_allow_html=True)
    
    # The file uploader is rendered inside the styled box
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])
    
    st.markdown("""
        <p style="color: #64748b; font-size: 12px; margin-top: 15px;">JPG, PNG, WEBP up to 10MB</p>
        <div style="background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.1); border-radius: 8px; padding: 10px; margin-top: 20px; font-size: 13px; color: #10b981; display: inline-flex; align-items: center; gap: 8px;">
            🔒 Your image is private and secure
        </div>
    </div></div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: white; font-size: 18px; margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px;">⭐ Prediction Result</h3>', unsafe_allow_html=True)
    
    if uploaded_file is None:
        # Placeholder state
        st.markdown("""
        <div style="height: 300px; display: flex; align-items: center; justify-content: center; color: #64748b; border-radius: 16px; background-color: rgba(0,0,0,0.2);">
            Upload an image to see the result
        </div>
        """, unsafe_allow_html=True)
    else:
        # Prediction state
        img_col, text_col = st.columns([1, 1])
        
        with img_col:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
        with text_col:
            st.markdown('<div style="text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">', unsafe_allow_html=True)
            
            with st.spinner("Analyzing..."):
                try:
                    temp_path = "temp_upload.jpg"
                    image.convert('RGB').save(temp_path)
                    pred_class, confidence = predict_image(model, temp_path)
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    is_dog = pred_class == "Dog"
                    color_hex = "#3b82f6" if is_dog else "#a855f7"
                    icon = "🐶" if is_dog else "🐱"
                    
                    st.markdown(f"""
                        <div style="font-size: 60px; margin-bottom: -10px;">{icon}</div>
                        <h2 style="color: {color_hex}; font-size: 3.5rem; font-weight: 800; margin: 0;">{pred_class}</h2>
                        <div class="result-badge">
                            ✓ This is a {pred_class}!
                        </div>
                        
                        <div style="margin-top: 30px;">
                            <div style="color: #94a3b8; font-size: 14px; margin-bottom: 5px;">Confidence Score</div>
                            <div style="color: {color_hex}; font-size: 2.5rem; font-weight: 800; line-height: 1;">{confidence:.2f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(int(confidence) / 100)
                    
                    st.markdown(f'<div style="color: #94a3b8; font-size: 13px; margin-top: 15px;">This image is most likely a <span style="color: {color_hex}; font-weight: 600;">{pred_class}</span>.</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

# --- FEATURE CARDS ---
st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon bg-purple">🎯</div>
        <div class="feature-text">
            <h4>High Accuracy</h4>
            <p>Trained on thousands of images for accurate predictions.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with f_col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon bg-blue">⚡</div>
        <div class="feature-text">
            <h4>Lightning Fast</h4>
            <p>Get results in just seconds with our optimized AI model.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with f_col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon bg-green">🛡️</div>
        <div class="feature-text">
            <h4>100% Secure</h4>
            <p>Your images are safe with us. We never store or share your data.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with f_col4:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon bg-pink">🧠</div>
        <div class="feature-text">
            <h4>AI Powered</h4>
            <p>Built with advanced deep learning technology.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class="custom-footer">
    Built with ❤️ using AI by <span>Priya Jain</span>
</div>
""", unsafe_allow_html=True)
