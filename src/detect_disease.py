import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('models/finetuned_model_v2.h5')

# Define class names to match test set directories
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']

def detect_disease(image):
    # Ensure image is in BGR format (from OpenCV) and convert to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img = image
    # Resize and normalize
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    # Expand dimensions for model prediction
    img = np.expand_dims(img, axis=0)
    # Predict
    pred = model.predict(img)
    class_idx = np.argmax(pred[0])
    confidence = pred[0][class_idx]
    return class_names[class_idx], confidence

if __name__ == "__main__":
    # Test with a local image file
    test_image_path = "data/processed/test/Bacterial Leaf Blight/sample_image.jpg"
    test_image = cv2.imread(test_image_path)
    if test_image is not None:
        result, conf = detect_disease(test_image)
        print(f"Detected: {result} (Confidence: {conf:.2f})")
    else:
        print(f"Failed to load test image: {test_image_path}")