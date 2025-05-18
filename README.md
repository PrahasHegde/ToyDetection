## All methods necessary for training and testing the model are listed in `main.py`

## TRAINING
Training weights are saved to: "runs/detect/train[specific model number]/weights/"

### Option 1: Run from main:
Execute the training function in `main.py`

### Option 2: Run from CLI:
```bash
yolo task=detect mode=train model=yolov8s.pt data=train.yaml epochs=100 imgsz=640 batch=8 device=0
```

# TESTING
The recordings of the live detection are saved to: "runs/detect/webcam_pred[specific model number]/"

## Testing (Live Detection)
To test the model on a webcam (or video/image), either use the function in `main.py` or run the following CLI command:
```bash
yolo task=detect mode=predict model=runs/detect/train8/weights/best.pt source=1
```


# VISUALIZATION 
# Train/Val statistics/graphs are stored in: 
runs/detect/train[<specific model number>]/results.png

