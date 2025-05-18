#RUN THIS FOR TRAINING-->
python train.py --img 416 --batch 8 --epochs 100 --data d.yaml --weights yolov5s.pt --name toy_detector

cam.py --> for webcam detection

#TEST THE MODEL
python detect.py --weights runs/train/toy_detector/weights/best.pt --img 640 --source path/to/images/

results --> runs/detect/exp/


#TEST ON WEBCAM 
python detect.py --weights runs/train/toy_detector/weights/best.pt --img 640 --source 0

#EVALUATION WITH TEST SET
python val.py --weights runs/train/toy_detector/weights/best.pt --data d.yaml --task test


#VISUALIZATION-->
runs/train/toy_detector/results.png

#ALL MODEL OUTPUTS , GRAPHS ARE IN --> yolov5\runs\train\toy_detector5

# YOLO11 branch
use the weights in runs/detect/train8/weights

# For Yolov11
yolo task=detect mode=predict model=runs/detect/train8/weights/best.pt source=1     <--- for detect mode
yolo task=detect mode=train model=yolov11.pt data=train.yaml epochs=100 imgsz=640 batch=8 device=0  <--- for training on GPU 
