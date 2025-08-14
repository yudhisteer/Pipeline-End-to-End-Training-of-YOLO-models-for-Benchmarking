#!/usr/bin/env python3
"""
Simplified YOLO training entry point for SageMaker.
Handles config.yaml with fallback to command-line arguments.
"""

import os
import sys
from ultralytics import YOLO

from utils import (
    generate_model_metrics, 
    send_metrics_to_cloudwatch, 
    save_metrics_for_pipeline,
    load_training_config,
    parse_training_args
)


def main():
    """Main training function for SageMaker."""
    print("Starting YOLO Training...")
    
    # Load configuration and parse arguments
    config = load_training_config()
    args = parse_training_args(config)
    
    # Get SageMaker paths
    input_data_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    
    print(f"Training data: {input_data_dir}")
    print(f"Model output: {model_dir}")
    print(f"Model: {args.model_name} | Epochs: {args.epochs} | Batch: {args.batch_size}")
    
    # Find data.yaml
    data_yaml_path = os.path.join(input_data_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        # Look in subdirectories
        for subdir in ["yolo_dataset", "train"]:
            alt_path = os.path.join(input_data_dir, subdir, "data.yaml")
            if os.path.exists(alt_path):
                data_yaml_path = alt_path
                break
        else:
            raise FileNotFoundError(f"data.yaml not found in {input_data_dir}")
    
    print(f"Dataset config: {data_yaml_path}")
    
    # Initialize and train model
    model = YOLO(args.model_name)
    print("Starting training...")
    
    results = model.train(
        data=data_yaml_path,
        imgsz=args.image_size,
        epochs=args.epochs,
        batch=args.batch_size,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        cos_lr=args.cos_lr,
        optimizer=args.optimizer,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        dropout=args.dropout,
        amp=args.amp,
        project=model_dir,
        name="",
        exist_ok=True
    )
    
    print("Training completed!!!")
    
    # Generate metrics and export model
    try:
        generate_model_metrics(results, model_dir)
        
        # Send metrics to CloudWatch
        output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
        metrics = save_metrics_for_pipeline(model_dir, output_dir)
        send_metrics_to_cloudwatch(metrics.get('recall', 0.0), metrics.get('map_50', 0.0))
        print("Metrics sent to CloudWatch")
        
    except Exception as e:
        print(f"Metrics processing failed: {e}")
    
    # Export model
    try:
        model.export(format="onnx", dynamic=True)
        print(f"Model exported to: {model_dir}")
    except Exception as e:
        print(f"Model export failed: {e}")
    
    print("All done!")

if __name__ == "__main__":
    main()
