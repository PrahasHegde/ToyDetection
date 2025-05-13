import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

#Set paths
input_path = r"C:\Users\hegde\OneDrive\Desktop\ToyDetection\valid"
images_folder = os.path.join(input_path, "images")
labels_folder = os.path.join(input_path, "labels")
json_path = os.path.join(input_path, "_annotations.coco.json")

# Ensure output folders exist
os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

#Load COCO annotations
with open(json_path, "r") as f:
    data = json.load(f)

#Build helper maps
filename_to_img = {img['file_name']: img for img in data['images']}
image_id_to_annotations = defaultdict(list)
for ann in data['annotations']:
    image_id_to_annotations[ann['image_id']].append(ann)

# Start processing
count = 0
for file_name, img_data in filename_to_img.items():
    src_path = os.path.join(images_folder, file_name)
    new_img_name = f"img{count}.jpg"
    dst_path = os.path.join(images_folder, new_img_name)

    #Check if image file exists
    if not os.path.exists(src_path):
        print(f"Image not found: {src_path}")
        continue

    #Copy and rename image
    shutil.copy(src_path, dst_path)

    img_id = img_data['id']
    img_w = img_data['width']
    img_h = img_data['height']
    anns = image_id_to_annotations.get(img_id, [])

    #Write YOLO labels
    label_file_path = os.path.join(labels_folder, f"img{count}.txt")
    with open(label_file_path, "w") as f:
        for ann in anns:
            cat_id = ann['category_id'] - 1  # YOLO expects class IDs to start at 0
            x, y, w, h = ann['bbox']
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w /= img_w
            h /= img_h
            f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f" Processed: {new_img_name}")
    count += 1

print("All conversions complete! YOLO labels saved to:", labels_folder)
