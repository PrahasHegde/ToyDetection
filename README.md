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
