import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def get_contour(ann_values, img_width, img_height):
    """Get the bounding box coordinates from the YOLOv8 pose
    format for one object (from a single row of the annotation file).

    Parameters
    ----------
    ann_values : list
        list with values from the YOLOv8 pose format for one object.
        (see docs).
    img_height : int
        image height in pixels
    img_width : int
        image width in pixels

    Returns
    -------
    keypoints_coords: list
        list of keypoints coords, each element of this list is
        tuple in form of (x, y, v), where
        x - coordinate of keypoint in pixels of image
        y - coordinate of keypoint in pixels of image
        v - status of visibility from MS COCO dataset
        v=0: not labeled (in which case x=y=0),
        v=1: labeled but not visible,
        and v=2: labeled and visible.
    """
    cnt_xs = ann_values[5::2]
    cnt_ys = ann_values[6::2]

    cnt = []
    for x, y in zip(cnt_xs, cnt_ys):
        x, y = unnormalize_coords(x, y, img_width, img_height)
        cnt.append([[x,y]])

    return np.array(cnt, np.int32)


def get_bbox_coco_coords(img_height, img_width, ann_values):
    """Get the bounding box coordinates from the YOLOv8 pose
    format for one object (from a single row of the annotation file).

    Parameters
    ----------
    img_height : int
        image height in pixels
    img_width : int
        image width in pixels
    ann_values : list
        list with values from the YOLOv8 pose format for one object.
        (see docs).

    Returns
    -------
    list
        list with bbox coordinates in MS COCO format
    """
    # YOLO bbox format normalized [x_center, y_center, width, height]
    x1, y1, x2, y2 = ann_values[1:5]
    x_min, y_min = unnormalize_coords(x1, y1, img_width, img_height)
    width, height = unnormalize_coords(x2, y2, img_width, img_height)

    # MS COCO bbox format [x_min, y_min, width, height]
    bbox_coords = (x_min, y_min, width, height)

    return bbox_coords


def unnormalize_coords(x, y, width, height):
    """Get the original image coordinates from the normalized
    coordinates.

    Parameters
    ----------
    x : float
        normalized coordinate on the X axis
    y : float
       normalized coordinate on the Y axis
    width : int
        image width in pixels
    height : int
        image height in pixels

    Returns
    -------
    x, y
       original coordinates x, y in pixels of the image.
    """
    x = int(x * width)
    y = int(y * height)
    return x, y

def visualize_annotations(image_path, annotation_path, output_img_path=None):
    """A function for visualizing markup from a text file in YOLOv8-pose
    format for three points on the image.
    It is useful for checking the correctness of the labeling.

    Parameters
    ----------
    image_path : str
        path to image from YOLOv8 pose dataset
    annotation_path : str
        path to text with labeling of keypoints from YOLOv8 pose dataset
    output_img_path : str, optional
        path to save output image with keypoint visualization, by default None
    """
    # Load the image
    image = cv2.imread(image_path)

    # Get the dimensions of the image
    img_height, img_width, _ = image.shape

    # Read annotations
    with open(annotation_path, "r") as f:
        annotations = f.readlines()

    for ann in annotations:
        # Parse annotation, this will need to be adjusted based on your format
        ann_values = list(map(float, ann.split()))

        # Get bbox coordinates
        bbox_coords = get_bbox_coco_coords(img_height, img_width, ann_values)

        # Get the key points coordinates
        cnt = get_contour(ann_values, img_width, img_height)

        # Draw bounding box
        x, y, w, h = bbox_coords
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw key points
        cv2.drawContours(image, [cnt], 0, (0,255,0),2)

    # Display the image
    if output_img_path is None:
        output_img_path = "viz.jpg"
   
    # cv2.namedWindow("viz", cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow("viz", image)
    # cv2.waitKey(0)

    # cv2.imwrite( output_img_path, image)
    plt.imshow(image[:,:,::-1])
    plt.show()

if __name__ == "__main__":
    # image_path = "/home/supercomputing/works/chloasma/datasets/coco8/images/train/000000000009.jpg"
    # annotation_path = "/home/supercomputing/works/chloasma/datasets/coco8/labels/train/000000000009.txt"

    # # Visualize the keypoints labeling
    # visualize_annotations(image_path, annotation_path)

    train_dir = "dataset/yolo_dataset/train/images"
    label_dir = "dataset/yolo_dataset/train/labels"
    for file in os.listdir(train_dir)[:10]:
        visualize_annotations(os.path.join(train_dir,file),os.path.join(label_dir,file.replace(".jpg",".txt")))