import os
import random
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# Set paths
coco_annotation_file = 'dataset/Bull Pig toy dataset.v2i.coco/test/_annotations.coco.json'
image_dir = 'dataset/Bull Pig toy dataset.v2i.coco/test/'

# Initialize COCO API
coco = COCO(coco_annotation_file)

# Get all category names and IDs
cat_ids = coco.getCatIds()
categories = coco.loadCats(cat_ids)
category_names = [cat['name'] for cat in categories]
print("Categories:", category_names)

# Get all image IDs
img_ids = coco.getImgIds()
random_img_id = random.choice(img_ids)
img_info = coco.loadImgs(random_img_id)[0]

# Load image
img_path = os.path.join(image_dir, img_info['file_name'])
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get annotations for this image
ann_ids = coco.getAnnIds(imgIds=img_info['id'], iscrowd=False)
annotations = coco.loadAnns(ann_ids)

# Plot image with annotations
plt.figure(figsize=(10, 8))
plt.imshow(image)
ax = plt.gca()

for ann in annotations:
    bbox = ann['bbox']
    cat_id = ann['category_id']
    cat_name = coco.loadCats(cat_id)[0]['name']
    
    x, y, w, h = bbox
    rect = plt.Rectangle((x, y), w, h, fill=False, color='red', linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y - 5, cat_name, color='white', fontsize=10,
            bbox=dict(facecolor='red', alpha=0.5))

plt.axis('off')
plt.title(f"Image ID: {img_info['id']}")
plt.show()
