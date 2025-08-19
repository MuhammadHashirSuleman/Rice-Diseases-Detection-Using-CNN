import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from detect_disease import detect_disease

# Define test directory and class names
test_dir = "data/processed/test"
class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']

# Create test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=9,  # Load 9 images for 3x3 grid
    class_mode='categorical',
    shuffle=True
)

# Visualize predictions
def visualize_predictions():
    imgs, _ = next(test_generator)  # Get one batch of 9 images
    fig = plt.figure(figsize=(15, 15))
    columns = 3
    rows = 3
    for i in range(columns * rows):
        fig.add_subplot(rows, columns, i + 1)
        img = imgs[i]
        # Predict using detect_disease
        predicted_class, confidence = detect_disease(img)
        # Display image and prediction
        plt.imshow(img)
        plt.text(10, 30, f"{predicted_class}\n(Conf: {confidence:.2f})", 
                 color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')  # Save the figure
    plt.show()

if __name__ == "__main__":
    visualize_predictions()