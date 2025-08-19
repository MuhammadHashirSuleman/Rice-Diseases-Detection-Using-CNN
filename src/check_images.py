import os
import cv2
from tqdm import tqdm

train_dir = "data/processed/train"
bad_images = []

for class_name in ['Bacterial Leaf Blight', 'Brown Spot', 'Leaf Smut']:
    class_path = os.path.join(train_dir, class_name)
    if os.path.exists(class_path):
        for img_name in tqdm(os.listdir(class_path), desc=f"Checking {class_name}"):
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    bad_images.append(img_path)
            except Exception as e:
                bad_images.append(img_path)
                print(f"Error with {img_path}: {e}")

if bad_images:
    print("Found bad images:", bad_images)
    with open("bad_images.txt", "w") as f:
        for img in bad_images:
            f.write(img + "\n")
else:
    print("No bad images found.")