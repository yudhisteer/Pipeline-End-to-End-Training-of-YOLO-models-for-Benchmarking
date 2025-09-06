import os
from ultralytics import YOLO
import boto3
from rich import print


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
    # TODO: use config file instead of env variables
    input_data_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    
    # Get model configuration from config
    model_name = config.get('training', {}).get('model_name')
    models_dir = config.get('training', {}).get('models_dir')
    
    print(f"Training data: {input_data_dir}")
    print(f"Model output: {model_dir}")
    print(f"Model name: {model_name}")
    print(f"Models directory: {models_dir}")

    # Download model from S3 to local storage only
    model_s3_path = f"{models_dir.rstrip('/')}/{model_name}"
    local_model_path = f"/tmp/{model_name}"
    
    print(f"Downloading YOLO model from S3: {model_s3_path}")
    print(f"Local destination: {local_model_path}")
    
    # Download model using boto3
    s3_client = boto3.client('s3')
    
    # Parse S3 URL
    s3_bucket = model_s3_path.split('/')[2]
    s3_key = '/'.join(model_s3_path.split('/')[3:])
    
    print(f"S3 bucket: {s3_bucket}")
    print(f"S3 key: {s3_key}")
    
    s3_client.download_file(s3_bucket, s3_key, local_model_path)
    print(f"Model downloaded successfully to: {local_model_path}")
    
    # Load model from local path
    model = YOLO(local_model_path)
    print(f"Model loaded successfully from local file")
    
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
    
    print(f"DEBUG: Dataset config: {data_yaml_path}")
    
    # Validate data.yaml content and dataset structure
    try:
        import yaml
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        print(f"DEBUG: data.yaml content:")
        print(f"  train: {data_config.get('train', 'NOT_FOUND')}")
        print(f"  val: {data_config.get('val', 'NOT_FOUND')}")
        print(f"  nc: {data_config.get('nc', 'NOT_FOUND')}")
        print(f"  names: {data_config.get('names', 'NOT_FOUND')}")
        
        # Check if train and val directories exist
        train_path = os.path.join(input_data_dir, data_config.get('train', ''))
        val_path = os.path.join(input_data_dir, data_config.get('val', ''))
        
        print(f"DEBUG: Checking dataset paths:")
        print(f"  Train images path: {train_path} - exists: {os.path.exists(train_path)}")
        print(f"  Val images path: {val_path} - exists: {os.path.exists(val_path)}")
        
        if os.path.exists(train_path):
            train_images = [f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  Train images count: {len(train_images)}")
        
        if os.path.exists(val_path):
            val_images = [f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"  Val images count: {len(val_images)}")
            
    except Exception as e:
        print(f"DEBUG: Error validating dataset: {e}")
    
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

    print("Hyperparameters used for training:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    
    # Validate critical hyperparameters
    lr0 = hyperparams.get('lr0', 0.01)
    epochs = hyperparams.get('epochs', 10)
    
    if lr0 > 0.05:
        print(f"WARNING: Learning rate {lr0} is very high. Consider using 0.01 or lower for YOLO.")
    if epochs < 5:
        print(f"WARNING: Only {epochs} epochs may be insufficient for meaningful training.")
    if 'val' not in str(data_config.get('val', '')):
        print(f"WARNING: Validation path may be incorrect: {data_config.get('val', 'NOT_SET')}")
    
    print("Starting training...")
    
    # Train model
    results = model.train(**hyperparams)
    print("Training completed!!!")
    
    # Generate metrics and export model
    try:
        generate_model_metrics(results, model_dir)
        
        # Send metrics to CloudWatch
        # TODO: need to check if this is true
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
