import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_model():
    # Define test directory and create results folder
    test_dir = "data/processed/test"
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Define class names
    class_names = ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']

    # Create test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )

    # Load the model
    try:
        model = tf.keras.models.load_model('models/finetuned_model_v2.h5')
    except:
        print("Failed to load finetuned_model_v2.h5, trying finetuned_model.h5")
        try:
            model = tf.keras.models.load_model('models/finetuned_model.h5')
        except:
            print("Failed to load finetuned_model.h5, trying trained_model.h5")
            model = tf.keras.models.load_model('models/trained_model.h5')

    # Evaluate model
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Generate predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    # Compute classification report
    report = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)
    print(classification_report(true_classes, predicted_classes, target_names=class_names))

    # Save metrics to text file
    with open(os.path.join(results_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
        f.write("-" * 65 + "\n")
        for cls in class_names:
            f.write(f"{cls:<25} {report[cls]['precision']:<10.2f} {report[cls]['recall']:<10.2f} "
                    f"{report[cls]['f1-score']:<10.2f} {int(report[cls]['support']):<10}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Macro Avg':<25} {report['macro avg']['precision']:<10.2f} {report['macro avg']['recall']:<10.2f} "
                f"{report['macro avg']['f1-score']:<10.2f} {int(report['macro avg']['support']):<10}\n")
        f.write(f"{'Weighted Avg':<25} {report['weighted avg']['precision']:<10.2f} {report['weighted avg']['recall']:<10.2f} "
                f"{report['weighted avg']['f1-score']:<10.2f} {int(report['weighted avg']['support']):<10}\n")

    # Plot confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

    # Plot bar plots for precision, recall, and F1-score
    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        values = [report[cls][metric] for cls in class_names]
        plt.figure(figsize=(8, 5))
        bars = plt.bar(class_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title(f'{metric.capitalize()} per Class')
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f'{value:.2f}', ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{metric}_bar_plot.png'))
        plt.close()

if __name__ == "__main__":
    evaluate_model()