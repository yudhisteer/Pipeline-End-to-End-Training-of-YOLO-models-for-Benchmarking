"""
Simplified YOLO training entry point for SageMaker.
Handles config.yaml with fallback to command-line arguments.
"""

import os
from ultralytics import YOLO

from utils.utils_metrics import (
    generate_model_metrics, 
    send_metrics_to_cloudwatch, 
    save_metrics_for_pipeline,
)
from utils.utils_config import load_training_config


def main():

    print("Starting YOLO Training...")
    
    # Load configuration
    config = load_training_config()
    
    # Get SageMaker paths
    input_data_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    
    # Get model name from config
    model_name = config.get('training', {}).get('model_name', 'yolo11n.pt') if config else 'yolo11n.pt'
    
    print(f"Training data: {input_data_dir}")
    print(f"Model output: {model_dir}")
    print(f"Model: {model_name}")
    
    # Find data.yaml
    data_yaml_path = os.path.join(input_data_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        # Look in subdirectories
        for subdir in ["yolo-dataset", "train"]:
            alt_path = os.path.join(input_data_dir, subdir, "data.yaml")
            if os.path.exists(alt_path):
                data_yaml_path = alt_path
                break
        else:
            raise FileNotFoundError(f"data.yaml not found in {input_data_dir}")
    
    print(f"Dataset config: {data_yaml_path}")
    
    # Initialize and train model
    model = YOLO(model_name)
    print("Starting training...")

    # Build hyperparameters directly from config
    hyperparams = {
        "data": data_yaml_path,
        "project": model_dir,
        "name": "",
        "exist_ok": True
    }
    
    # Read training.hyperparams directly from config
    training_hyperparams = config['training']['hyperparams']
        
    # Add all hyperparams from config directly
    hyperparams.update(training_hyperparams)

    print("Hyperparameters for training:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # Train model
    results = model.train(**hyperparams)
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
