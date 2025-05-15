from ultralytics import YOLO
# from ultralytics import 

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="train.yaml",  # Path to dataset YAML (must be segmentation-compatible)
    epochs=1000,  # Number of training epochs
    imgsz=640,  # Image size
    device=0,  # GPUs to use (or "cpu" for CPU training)
    batch=16,  # Adjust batch size based on GPU memory
    workers=4,  # Number of dataloader workers
    optimizer="AdamW",  # AdamW optimizer (optional, can use "SGD")
    lr0=0.001,  # Initial learning rate
    patience=50,  # Early stopping patience
    seed=42,  # Random seed for reproducibility
    verbose=True, # Display training progress
    multi_scale = True,
    pretrained = True,
    single_cls = False,
    cos_lr=True,
    box = 15,
)

print("train finished!!!!!!!!!!!!!!!")