# src/detect_plate.py
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("models/yolov8n_plate.pt")  # or "yolov8n.pt" then fine-tune

def detect_plates(frame, conf_thresh=0.3):
    results = model.predict(source=frame, conf=conf_thresh, imgsz=640, verbose=False)
    plates = []
    # results[0].boxes -> xyxy, conf, cls
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            plates.append({"box": (x1,y1,x2,y2), "conf": conf, "cls": cls})
    return plates

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        if not ret: break
        plates = detect_plates(frame)
        for p in plates:
            x1,y1,x2,y2 = p['box']
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow("ANPR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
