import os
from pycocotools.coco import COCO
import shutil


def convert_coco_dataset_to_yolo(sub_dir):
    # Paths
    coco_annotation_file = f'dataset/Bull Pig toy dataset.v2i.coco/{sub_dir}/_annotations.coco.json'  # path to your COCO JSON
    image_dir = f'dataset/Bull Pig toy dataset.v2i.coco/{sub_dir}'  # path to your COCO image directory
    output_dir = f'dataset/yolo_dataset/{sub_dir}/labels'  # where to save YOLO annotations
    output_img_dir = f'dataset/yolo_dataset/{sub_dir}/images/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_img_dir, exist_ok=True)

    # Load COCO data
    coco = COCO(coco_annotation_file)

    # Build category mapping: COCO category_id -> 0-based YOLO class_id
    categories = coco.loadCats(coco.getCatIds())
    cat_mapping = {cat['id']: i for i, cat in enumerate(categories)}

    # Convert each image
    img_ids = coco.getImgIds()
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        width, height = img_info['width'], img_info['height']
        file_name = os.path.splitext(img_info['file_name'])[0]
        label_name = file_name + '.txt'
        img_name = file_name + ".jpg"
        output_path = os.path.join(output_dir, label_name)
        shutil.copy(os.path.join(image_dir, img_name), output_img_dir)

        # Get annotations
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        with open(output_path, 'w') as f:
            for ann in anns:
                cat_id = ann['category_id']
                class_id = cat_mapping[cat_id]

                x, y, w, h = ann['bbox']
                # Convert to YOLO format (normalized center x, y, width, height)
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                w_norm = w / width
                h_norm = h / height

                f.write(f"{class_id - 1} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
