# import os
# import cv2
# import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm

# def preprocess_images(input_dir, output_dir, target_size=(224, 224)):
#     if not os.path.exists(input_dir):
#         raise FileNotFoundError(f"Input directory {input_dir} does not exist. Please add the dataset.")
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     datagen = ImageDataGenerator(
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         horizontal_flip=True,
#         zoom_range=0.2,
#         rescale=1./255
#     )
#     total_images = 0
#     for class_name in ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']:
#         class_input = os.path.join(input_dir, class_name)
#         if os.path.exists(class_name):
#             total_images += len(os.listdir(class_input))
#     with tqdm(total=total_images, desc="Preprocessing Images") as pbar:
#         for class_name in ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']:
#             class_input = os.path.join(input_dir, class_name)
#             class_output = os.path.join(output_dir, class_name)
#             if not os.path.exists(class_output):
#                 os.makedirs(class_output)
#             if os.path.exists(class_input):
#                 for img_name in os.listdir(class_input):
#                     img_path = os.path.join(class_input, img_name)
#                     img = cv2.imread(img_path)
#                     if img is not None:
#                         img = cv2.resize(img, target_size)
#                         img = np.expand_dims(img, axis=0)
#                         for batch in datagen.flow(img, batch_size=1, save_to_dir=class_output, save_prefix='aug', save_format='jpg'):
#                             break  # Generate one augmented image per original
#                     else:
#                         print(f"Failed to load {img_path}, skipping...")
#                     pbar.update(1)  # Update progress for each image
#     print(f"Preprocessing completed for {input_dir}.")

# def split_dataset(data_dir, train_dir, val_dir, test_dir, test_size=0.2, val_size=0.25):
#     if not os.path.exists(train_dir):
#         os.makedirs(train_dir, exist_ok=True)
#     if not os.path.exists(val_dir):
#         os.makedirs(val_dir, exist_ok=True)
#     if not os.path.exists(test_dir):
#         os.makedirs(test_dir, exist_ok=True)
#     total_images = 0
#     for class_name in ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']:
#         class_path = os.path.join(data_dir, class_name)
#         if os.path.exists(class_path):
#             total_images += len(os.listdir(class_path))
#     with tqdm(total=total_images, desc="Splitting Dataset") as pbar:
#         for class_name in ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']:
#             class_path = os.path.join(data_dir, class_name)
#             if not os.path.exists(class_path):
#                 continue  # Skip if class directory is missing
#             images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
#             if not images:
#                 print(f"No images found in {class_path}, skipping...")
#                 continue
#             train_val, test = train_test_split(images, test_size=test_size, random_state=42)
#             train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
#             for img in train: 
#                 os.replace(img, os.path.join(train_dir, class_name, os.path.basename(img)))
#                 pbar.update(1)
#             for img in val: 
#                 os.replace(img, os.path.join(val_dir, class_name, os.path.basename(img)))
#                 pbar.update(1)
#             for img in test: 
#                 os.replace(img, os.path.join(test_dir, class_name, os.path.basename(img)))
#                 pbar.update(1)
#     print(f"Dataset split completed for {data_dir}.")

# if __name__ == "__main__":
#     raw_dir = "data/raw/mendeley"
#     augmented_dir = "data/augmented"
#     processed_dir = "data/processed"
#     preprocess_images(raw_dir, augmented_dir)
#     split_dataset(augmented_dir, os.path.join(processed_dir, "train"), os.path.join(processed_dir, "validation"), os.path.join(processed_dir, "test"))

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def preprocess_images(input_dir, output_dir, target_size=(224, 224)):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist. Please add the dataset.")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        zoom_range=0.3,
        shear_range=0.2
    )
    total_images = 0
    for class_name in ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']:
        class_input = os.path.join(input_dir, class_name)
        if os.path.exists(class_input):
            total_images += len(os.listdir(class_input))
    with tqdm(total=total_images, desc="Preprocessing Images") as pbar:
        for class_name in ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']:
            class_input = os.path.join(input_dir, class_name)
            class_output = os.path.join(output_dir, class_name)
            if not os.path.exists(class_output):
                os.makedirs(class_output)
            if os.path.exists(class_input):
                for img_name in os.listdir(class_input):
                    img_path = os.path.join(class_input, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, target_size)
                        img = np.expand_dims(img, axis=0)
                        # Generate 5 augmented images for Leaf Smut, 1 for others
                        num_augmentations = 5 if class_name == 'Leaf Smut' else 1
                        for i in range(num_augmentations):
                            for batch in datagen.flow(img, batch_size=1, save_to_dir=class_output, save_prefix=f'aug_{i}', save_format='jpg'):
                                break
                    else:
                        print(f"Failed to load {img_path}, skipping...")
                    pbar.update(1)
    print(f"Preprocessing completed for {input_dir}.")

def split_dataset(data_dir, train_dir, val_dir, test_dir, test_size=0.2, val_size=0.25):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir, exist_ok=True)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir, exist_ok=True)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)
    total_images = 0
    for class_name in ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            total_images += len(os.listdir(class_path))
    with tqdm(total=total_images, desc="Splitting Dataset") as pbar:
        for class_name in ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']:
            class_path = os.path.join(data_dir, class_name)
            if not os.path.exists(class_path):
                continue
            images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
            if not images:
                print(f"No images found in {class_path}, skipping...")
                continue
            train_val, test = train_test_split(images, test_size=test_size, random_state=42)
            train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
            for img in train:
                os.replace(img, os.path.join(train_dir, class_name, os.path.basename(img)))
                pbar.update(1)
            for img in val:
                os.replace(img, os.path.join(val_dir, class_name, os.path.basename(img)))
                pbar.update(1)
            for img in test:
                os.replace(img, os.path.join(test_dir, class_name, os.path.basename(img)))
                pbar.update(1)
    print(f"Dataset split completed for {data_dir}.")

if __name__ == "__main__":
    raw_dir = "data/raw/mendeley"
    augmented_dir = "data/augmented"
    processed_dir = "data/processed"
    preprocess_images(raw_dir, augmented_dir)
    split_dataset(augmented_dir, os.path.join(processed_dir, "train"), os.path.join(processed_dir, "validation"), os.path.join(processed_dir, "test"))