import torch.serialization
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO

# --- THE FIX ---
# Manually add required classes to PyTorch's "safe list"
# This fixes the _pickle.UnpicklingError in PyTorch 2.6+
try:
    torch.serialization.add_safe_globals([
        DetectionModel,
        nn.modules.container.Sequential,
        nn.modules.conv.Conv2d,
        nn.modules.batchnorm.BatchNorm2d,
        nn.modules.activation.SiLU,
        nn.modules.pooling.MaxPool2d,
        nn.modules.upsampling.Upsample,
    ])
except AttributeError:
    # Handle older PyTorch versions that might not have this function
    print("Could not set add_safe_globals. This might be an older PyTorch version.")
    pass
# --- END FIX ---


if __name__ == '__main__':
    # 1. Load a pre-trained model
    #    This should work now
    print("Loading model...")
    model = YOLO('yolov8n.pt') 

    # 2. Start training
    print("Starting training...")
    results = model.train(
        data=r'C:\Users\Bernardo Carlo\Documents\python-training\dataset\data.yaml',
        epochs=50,
        imgsz=640,
        name='license_plate_run1',
        workers=0  # Set to 0 to avoid multiprocessing issues on Windows
    )

    print("Training complete! Model saved.")