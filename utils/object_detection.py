import numpy as np

from ultralytics import YOLO


model = YOLO("yolov8n.pt")


def detect_objects(frames):

    detected_objects = []

    for img in frames:

        frame_np = np.array(img)

        results = model(frame_np)

        names = results[0].names

        boxes = results[0].boxes

        for box in boxes:

            cls_id = int(box.cls[0])

            detected_objects.append(
                names[cls_id]
            )

    return list(set(detected_objects))