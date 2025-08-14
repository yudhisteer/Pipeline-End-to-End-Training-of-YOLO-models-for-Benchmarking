#!/usr/bin/env python3
"""
Minimal YOLO training entry point for SageMaker.
"""

import os
import sys
from ultralytics import YOLO

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import generate_model_metrics


def main():
    """Main training function for SageMaker."""
    
    # Get SageMaker environment paths
    input_data_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    
    # Get hyperparameters from SageMaker (passed from config.yaml via sagemaker_trainer.py)
    model_name = os.environ.get("SM_HPS_MODEL_NAME", "yolo11n.pt")
    image_size = int(os.environ.get("SM_HPS_IMAGE_SIZE", "640"))
    epochs = int(os.environ.get("SM_HPS_EPOCHS", "100"))
    batch_size = int(os.environ.get("SM_HPS_BATCH_SIZE", "16"))
    
    # Learning rate & optimization hyperparameters
    lr0 = float(os.environ.get("SM_HPS_LR0", "0.01"))
    lrf = float(os.environ.get("SM_HPS_LRF", "0.1"))
    momentum = float(os.environ.get("SM_HPS_MOMENTUM", "0.937"))
    weight_decay = float(os.environ.get("SM_HPS_WEIGHT_DECAY", "0.0005"))
    warmup_epochs = float(os.environ.get("SM_HPS_WARMUP_EPOCHS", "3.0"))
    cos_lr = os.environ.get("SM_HPS_COS_LR", "true").lower() == "true"
    optimizer = os.environ.get("SM_HPS_OPTIMIZER", "AdamW")
    
    # Data augmentation hyperparameters
    hsv_h = float(os.environ.get("SM_HPS_HSV_H", "0.015"))
    hsv_s = float(os.environ.get("SM_HPS_HSV_S", "0.7"))
    hsv_v = float(os.environ.get("SM_HPS_HSV_V", "0.4"))
    degrees = float(os.environ.get("SM_HPS_DEGREES", "10.0"))
    translate = float(os.environ.get("SM_HPS_TRANSLATE", "0.1"))
    scale = float(os.environ.get("SM_HPS_SCALE", "0.5"))
    fliplr = float(os.environ.get("SM_HPS_FLIPLR", "0.5"))
    mosaic = float(os.environ.get("SM_HPS_MOSAIC", "1.0"))
    mixup = float(os.environ.get("SM_HPS_MIXUP", "0.1"))
    
    # Loss function weights
    box = float(os.environ.get("SM_HPS_BOX", "7.5"))
    cls = float(os.environ.get("SM_HPS_CLS", "0.5"))
    dfl = float(os.environ.get("SM_HPS_DFL", "1.5"))
    label_smoothing = float(os.environ.get("SM_HPS_LABEL_SMOOTHING", "0.1"))
    
    # Advanced training options
    patience = int(os.environ.get("SM_HPS_PATIENCE", "50"))
    dropout = float(os.environ.get("SM_HPS_DROPOUT", "0.1"))
    amp = os.environ.get("SM_HPS_AMP", "true").lower() == "true"
    
    print(f"Input data directory: {input_data_dir}")
    print(f"Model directory: {model_dir}")
    print(f"Training with: {model_name}, epochs={epochs}, batch={batch_size}, image_size={image_size}")
    print(f"Learning rate: lr0={lr0}, lrf={lrf}, optimizer={optimizer}")
    print(f"Augmentation: hsv_h={hsv_h}, degrees={degrees}, fliplr={fliplr}")
    print(f"Loss weights: box={box}, cls={cls}, dfl={dfl}")
    
    # Initialize YOLO model
    model = YOLO(model_name)
    
    # Use data.yaml from training data directory (uploaded by SageMakerDataManager)
    # The complete dataset is now uploaded, so data.yaml should be in the root of training data
    data_yaml_path = os.path.join(input_data_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        # Fallback: look for data.yaml in subdirectories (backward compatibility)
        possible_paths = [
            os.path.join(input_data_dir, "yolo_dataset", "data.yaml"),
            os.path.join(input_data_dir, "train", "data.yaml"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_yaml_path = path
                break
        else:
            raise FileNotFoundError(f"data.yaml not found in {input_data_dir} or its subdirectories")
    
    print(f"Using data.yaml from: {data_yaml_path}")
    
    # Train the model with all hyperparameters
    results = model.train(
        data=data_yaml_path,
        imgsz=image_size,
        epochs=epochs,
        batch=batch_size,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        cos_lr=cos_lr,
        optimizer=optimizer,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        box=box,
        cls=cls,
        dfl=dfl,
        label_smoothing=label_smoothing,
        patience=patience,
        dropout=dropout,
        amp=amp,
        project=model_dir,
        name="",
        exist_ok=True
    )
    print(f"Training results: {results}")
    
    # Generate validation metrics for SageMaker Model Registry
    generate_model_metrics(results, model_dir)
    
    # Export model
    model.export(format="onnx", dynamic=True)
    print("Model exported to: ", model_dir)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
