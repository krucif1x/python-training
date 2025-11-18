from ultralytics import YOLO

# 1. Load your trained model
#    Make sure to use the correct path to YOUR 'best.pt' file
model = YOLO(r'runs/detect/license_plate_run1/weights/best.pt')

# 2. Define your source
#    This can be an image, video, or even a webcam feed (use source=0)
source_image = 'test_image.jpg'
# source_video = 'test_video.mp4'
# source_webcam = 0

# 3. Run inference
print("Running inference...")
# 'save=True' will save a copy of the image/video with boxes
# 'conf=0.4' sets the confidence threshold to 40%
results = model.predict(
    source=source_image, 
    save=True, 
    conf=0.4
)

print("Inference complete!")

# 4. (Optional) Loop through results and print details
for r in results:
    boxes = r.boxes  # Get the bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates
        conf = box.conf[0]           # Get confidence score
        cls = box.cls[0]             # Get class ID

        print(f"Detected: {model.names[int(cls)]} with {conf*100:.2f}% confidence.")
        print(f"Coordinates: [({x1:.0f}, {y1:.0f}), ({x2:.0f}, {y2:.0f})]")