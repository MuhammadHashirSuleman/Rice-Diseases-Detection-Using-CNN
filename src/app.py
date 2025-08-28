import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detect_disease import detect_disease
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Custom CSS for black and emerald green theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Main app styling */
    .stApp {
        background: #0a0a0a;
        font-family: 'Inter', sans-serif;
        color: #e6e6e6;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        color: #2ECC71;
        margin: 2.5rem 0 1rem;
        text-shadow: 0 0 15px rgba(46, 204, 113, 0.5);
        animation: pulse 3s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }

    /* Subtitle styling */
    .subtitle {
        color: #50C878;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 0.5px;
        opacity: 0.9;
    }

    /* Card container styling */
    .card {
        background: #1a1a1a;
        border: 1px solid #2ECC71;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(46, 204, 113, 0.15);
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(46, 204, 113, 0.25);
    }

    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, transparent, #2ECC71, transparent);
        animation: slide 4s infinite;
    }

    @keyframes slide {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    /* Upload zone styling */
    .upload-zone {
        border: 2px dashed #2ECC71;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(46, 204, 113, 0.05);
        transition: all 0.3s ease;
    }

    .upload-zone:hover {
        background: rgba(46, 204, 113, 0.1);
        border-color: #50C878;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2ECC71, #50C878) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.8rem 1.5rem !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(46, 204, 113, 0.3) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #50C878, #2ECC71) !important;
        box-shadow: 0 8px 20px rgba(46, 204, 113, 0.5) !important;
        transform: translateY(-2px) !important;
    }

    /* File uploader styling */
    .stFileUploader > div > div > div {
        background: #1a1a1a !important;
        border: 1px solid #2ECC71 !important;
        border-radius: 8px !important;
        color: #e6e6e6 !important;
    }

    /* Result text styling */
    .result-text {
        color: #2ECC71;
        font-size: 1.5rem;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }

    .confidence-text {
        color: #e6e6e6;
        font-size: 1.1rem;
        text-align: center;
        opacity: 0.85;
    }

    /* Section headers */
    .section-header {
        color: #2ECC71;
        font-size: 1.6rem;
        font-weight: 600;
        text-align: center;
        margin: 1.5rem 0 1rem;
        position: relative;
    }

    .section-header::after {
        content: '';
        position: absolute;
        bottom: -4px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 2px;
        background: #2ECC71;
        border-radius: 2px;
    }

    /* Feature icons */
    .feature-icon {
        font-size: 2.5rem;
        color: #2ECC71;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* Info box styling */
    .info-box {
        background: rgba(46, 204, 113, 0.1);
        border-left: 3px solid #2ECC71;
        border-radius: 0 8px 8px 0;
        padding: 1rem;
        margin: 1rem 0;
        color: #e6e6e6;
        font-size: 0.95rem;
    }

    /* Image container styling */
    .image-container {
        border: 2px solid #2ECC71;
        border-radius: 12px;
        padding: 8px;
        background: #1a1a1a;
        box-shadow: 0 8px 20px rgba(46, 204, 113, 0.15);
        margin: 1rem 0;
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background: #2ECC71 !important;
        border-radius: 8px !important;
    }

    /* Spinner styling */
    .stSpinner > div {
        border-color: #2ECC71 transparent #2ECC71 transparent !important;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: #0a0a0a !important;
        border-right: 1px solid #2ECC71 !important;
    }

    /* Stat card styling */
    .stat-card {
        background: #1a1a1a;
        border: 1px solid #2ECC71;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 12px rgba(46, 204, 113, 0.15);
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-title { font-size: 2.2rem; }
        .subtitle { font-size: 1rem; }
        .section-header { font-size: 1.4rem; }
        .result-text { font-size: 1.3rem; }
        .confidence-text { font-size: 1rem; }
        .card { padding: 1.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<h1 class="main-title">🌾 Rice Leaf Disease Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Crop Health Analysis</p>', unsafe_allow_html=True)

# Upload section
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-icon">📸</div>', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">Upload Rice Leaf Image</h3>', unsafe_allow_html=True)
    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "png", "jpeg"],
        help="Upload a clear image of a rice leaf for disease detection"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Image and results display
if uploaded_file:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📷 Uploaded Image</h3>', unsafe_allow_html=True)
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded Rice Leaf", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🔍 Analysis Results</h3>', unsafe_allow_html=True)
        with st.spinner('Analyzing rice leaf...'):
            result, confidence = detect_disease(img)
        st.markdown(f'<div class="result-text">🦠 Detected: {result}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence-text">📊 Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
        st.progress(confidence / 100.0)
        if confidence > 80:
            st.markdown('<div class="info-box">✅ High confidence detection - Results are reliable</div>', unsafe_allow_html=True)
        elif confidence > 60:
            st.markdown('<div class="info-box">⚠️ Moderate confidence - Consider additional analysis</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">❌ Low confidence - Image quality may need improvement</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Annotated analysis
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">🎯 Annotated Analysis</h3>', unsafe_allow_html=True)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 5), facecolor='#1a1a1a')
    ax = plt.subplot(111)
    ax.set_facecolor('#1a1a1a')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.text(10, 30, f"{result}\nConfidence: {confidence:.1f}%",
             color='#2ECC71', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='#1a1a1a', alpha=0.8, edgecolor='#2ECC71', linewidth=1.5))
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# Test set predictions
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-icon">📊</div>', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">Test Set Analysis</h3>', unsafe_allow_html=True)
    if st.button("🚀 Show Test Set Predictions", key="test_predictions"):
        with st.spinner('Loading test set predictions...'):
            try:
                test_dir = "data/processed/test"
                test_datagen = ImageDataGenerator(rescale=1./255)
                test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    target_size=(224, 224),
                    batch_size=9,
                    class_mode='categorical',
                    shuffle=True
                )
                imgs, _ = next(test_generator)
                plt.style.use('dark_background')
                fig = plt.figure(figsize=(12, 12), facecolor='#1a1a1a')
                columns, rows = 3, 3
                for i in range(columns * rows):
                    ax = fig.add_subplot(rows, columns, i + 1)
                    ax.set_facecolor('#1a1a1a')
                    img = imgs[i]
                    predicted_class, confidence = detect_disease(img)
                    plt.imshow(img)
                    plt.text(10, 30, f"{predicted_class}\nConf: {confidence:.1f}%",
                             color='#2ECC71', fontsize=10, fontweight='bold',
                             bbox=dict(facecolor='#1a1a1a', alpha=0.8, edgecolor='#2ECC71', linewidth=1))
                    plt.axis('off')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown('<div class="info-box">✅ Test set predictions completed successfully!</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading test set: {str(e)}")
                st.markdown('<div class="info-box">❌ Please ensure test data directory exists and is accessible</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# About section
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📋 About This Tool</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-icon">🎯</div>', unsafe_allow_html=True)
        st.markdown('**Accuracy**')
        st.markdown('High-precision AI detection')
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-icon">⚡</div>', unsafe_allow_html=True)
        st.markdown('**Speed**')
        st.markdown('Real-time analysis')
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-icon">🌱</div>', unsafe_allow_html=True)
        st.markdown('**Impact**')
        st.markdown('Protecting crop health')
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)