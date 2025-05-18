from ultralytics import YOLO


def start_live_detection(path_to_weights: str, source=1, show=True, save=True, save_txt=False, save_conf=False, name="webcam_pred", vid_stride=10, show_labels=True):
    model = YOLO(path_to_weights)

    results = model.predict(
        source=source,
        show=show,
        save=save,
        save_txt=save_txt,
        save_conf=save_conf,
        name=name,
        vid_stride=vid_stride,
        show_labels=show_labels
    )
    for r in results:
        r.save()
