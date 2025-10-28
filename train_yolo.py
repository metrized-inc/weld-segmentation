from ultralytics import YOLO
import torch

def train_yolo_segmentation():
    """Train YOLOv8 segmentation model on custom dataset"""
    
    # Verify CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Load a pretrained YOLOv8 segmentation model
    # Options: yolov8n-seg.pt (nano), yolov8s-seg.pt (small), 
    #          yolov8m-seg.pt (medium), yolov8l-seg.pt (large), yolov8x-seg.pt (xlarge)
    model = YOLO('yolov8s-seg.pt')  # Start with nano for faster training
    
    # Train the model
    results = model.train(
        data='yolo_dataset/data.yaml',  # Path to dataset config
        epochs=100,                      # Number of epochs
        imgsz=640,                       # Image size
        batch=16,                        # Batch size (adjust based on GPU memory)
        device=0,                        # GPU device (0 for first GPU, 'cpu' for CPU)
        workers=8,                       # Number of worker threads
        patience=50,                     # Early stopping patience
        save=True,                       # Save checkpoints
        project='runs/segment',          # Project directory
        name='weld_segmentation',        # Experiment name
        exist_ok=True,                   # Overwrite existing experiment
        pretrained=True,                 # Use pretrained weights
        optimizer='Adam',                # Optimizer (Adam, SGD, AdamW)
        lr0=0.001,                        # Initial learning rate
        lrf=0.01,                        # Final learning rate factor
        momentum=0.937,                  # SGD momentum
        weight_decay=0.0005,             # Weight decay
        warmup_epochs=3,                 # Warmup epochs
        warmup_momentum=0.8,             # Warmup momentum
        box=7.5,                         # Box loss gain
        cls=0.5,                         # Class loss gain
        dfl=1.5,                         # DFL loss gain
        mosaic=1.0,                      # Mosaic augmentation probability
        mixup=0.0,                       # Mixup augmentation probability
        copy_paste=0.0,                  # Copy-paste augmentation probability
        degrees=0.0,                     # Rotation augmentation degrees
        translate=0.1,                   # Translation augmentation
        scale=0.5,                       # Scale augmentation
        shear=0.0,                       # Shear augmentation degrees
        perspective=0.0,                 # Perspective augmentation
        flipud=0.0,                      # Flip up-down probability
        fliplr=0.5,                      # Flip left-right probability
        hsv_h=0.015,                     # HSV-Hue augmentation
        hsv_s=0.7,                       # HSV-Saturation augmentation
        hsv_v=0.4,                       # HSV-Value augmentation
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nBest model saved to: runs/segment/weld_segmentation/weights/best.pt")
    print(f"Results saved to: runs/segment/weld_segmentation/")
    
    # Validate the model
    print("\nValidating model...")
    metrics = model.val()
    print(f"\nmAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    
    return model


if __name__ == "__main__":
    # Train the model
    model = train_yolo_segmentation()