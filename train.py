import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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
    print("Loading model...")
    model = YOLO('yolov8n.pt')  # You can also try 'yolov8s.pt' or 'yolov8m.pt' for better accuracy

    # 2. Start training with optimized parameters for license plate detection
    print("Starting training...")
    results = model.train(
        # Dataset
        data=r'C:\Users\Bernardo Carlo\Documents\python-training\dataset\data.yaml',
        
        # Training duration
        epochs=100,                # Set to 50
        patience=10,              # Early stopping if no improvement for 10 epochs
        
        # Image settings
        imgsz=640,                # Standard size, good for license plates
        
        # Batch size (adjust based on your GPU memory)
        batch=24,                 # *** CHANGED from 16 to 24 ***
        
        # Optimization
        optimizer='AdamW',        # Often better than SGD for small objects
        lr0=0.001,                # Initial learning rate (lower for fine-tuning)
        lrf=0.01,                 # Final learning rate (as fraction of lr0)
        momentum=0.937,           # Momentum for SGD
        weight_decay=0.0005,      # L2 regularization
        
        # Data augmentation (important for license plates in various conditions)
        degrees=10.0,             # Rotation augmentation (±10 degrees)
        translate=0.1,            # Translation augmentation
        scale=0.5,                # Scale augmentation
        shear=5.0,                # Shear augmentation
        perspective=0.0001,       # Perspective augmentation
        flipud=0.0,               # No vertical flip (license plates shouldn't be upside down)
        fliplr=0.5,               # 50% horizontal flip
        mosaic=1.0,               # Mosaic augmentation
        mixup=0.1,                # Mixup augmentation (10% of the time)
        
        # Color augmentation (for different lighting conditions)
        hsv_h=0.015,              # Hue augmentation
        hsv_s=0.7,                # Saturation augmentation
        hsv_v=0.4,                # Value (brightness) augmentation
        
        # Performance
        workers=3,                
        device=0,                 # Use GPU 0 (your RTX 3060)
        
        # Validation
        val=True,                 # Validate during training
        
        # Saving
        save=True,                # Save checkpoints
        plots=False,              # Disable plots to avoid Pillow errors
        
        # Output
        name='license_plate_run1',
        exist_ok=False,           # Don't overwrite existing runs
        
        # Misc
        verbose=True,             # Print detailed logs
        seed=42,                  # For reproducibility
    )

    print("Training complete! Model saved.")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")