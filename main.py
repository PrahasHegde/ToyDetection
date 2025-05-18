from coco_to_yolo import convert_coco_dataset_to_yolo
from live_detect import start_live_detection
from train import train_model

if __name__ == "__main__":
    #convert_coco_dataset_to_yolo("valid")
    #train_model()
    start_live_detection(
        path_to_weights="runs/detect/train8/weights/best.pt",
        source=1,
        name="webcam_pred",
        vid_stride=10
    )
