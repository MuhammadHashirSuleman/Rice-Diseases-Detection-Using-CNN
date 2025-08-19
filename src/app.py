import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detect_disease import detect_disease
from tensorflow.keras.preprocessing.image import ImageDataGenerator

st.title("Rice Plant Disease Detection")

# File uploader for single image prediction
uploaded_file = st.file_uploader("Upload Rice Leaf Image", type=["jpg", "png"])
if uploaded_file:
    # Read and decode the uploaded image
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    # Get prediction
    result, confidence = detect_disease(img)
    
    # Display original image
    st.image(img, caption="Uploaded Image", width=300)
    st.write(f"Detected: {result} (Confidence: {confidence:.2f})")
    
    # Visualize prediction on the image
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.text(10, 30, f"{result}\n(Conf: {confidence:.2f})", 
             color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

# Optional: Show test set predictions
if st.button("Show Test Set Predictions"):
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
    fig = plt.figure(figsize=(15, 15))
    columns = 3
    rows = 3
    for i in range(columns * rows):
        fig.add_subplot(rows, columns, i + 1)
        img = imgs[i]
        predicted_class, confidence = detect_disease(img)
        plt.imshow(img)
        plt.text(10, 30, f"{predicted_class}\n(Conf: {confidence:.2f})", 
                 color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        plt.axis('off')
    plt.tight_layout()
    st.pyplot(fig)