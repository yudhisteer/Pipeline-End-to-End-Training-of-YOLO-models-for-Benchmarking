import os
import yaml
import logging
import shutil
from ultralytics import YOLO
import pandas as pd
import json
import boto3

# -----------------------
# CloudWatch Log Settings
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# SageMaker environment path settings
# -----------------------
def get_sagemaker_paths():
    try:
        input_data_dir = os.environ.get("SM_CHANNEL_TRAINING", "./data")
        model_dir = os.environ.get("SM_MODEL_DIR", "./model")
        config_dir = os.environ.get("SM_CHANNEL_CONFIG", "./config")
        output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "./output")
        return input_data_dir, config_dir, model_dir, output_dir
    except Exception as e:
        logger.error(f"❌Failed to get SageMaker environment paths: {e}")
        raise

# -----------------------
# Read Config file and log contents (no comments, one line)
# -----------------------
def load_config(config_path):
    if not os.path.isfile(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"❌Config file not found: {config_path}")
    try:
        # Read the file contents for log(remove comment lines and combine into one line)
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        filtered = " ".join(
            line.strip() for line in lines
            if line.strip() and not line.strip().startswith("#")
        )
        logger.info(f"💡 Config content: {filtered}")

        # Read Config file
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config, filtered

    except Exception as e:
        logger.error(f"❌Failed to load config file {config_path}: {e}", exc_info=True)
        raise

# -----------------------
# Model learning
# -----------------------
def train_model(config, data_dir, model_dir):
    try:
        logger.info(f"ℹ️Starting YOLO training with config: {config}")

        if "model_name" not in config:
            logger.error(f"❌Training failed: {e}", exc_info=True)
            raise ValueError("❌model_name is not in config")
        model_name = config["model_name"]
        model = YOLO(model_name)

        # Hyperparameters
        img_size = config.get("image_size", 640)
        epochs = config.get("epochs", 2)
        batch_size = config.get("batch_size", 2)

        model.train(
            data=os.path.join(data_dir, "data.yaml"),
            imgsz=img_size,
            epochs=epochs,
            batch=batch_size,
            project=model_dir,
            name="",
            exist_ok=True
        )
        return model
    except Exception as e:
        logger.error(f"❌Training failed: {e}", exc_info=True)
        raise

# -----------------------
# ONNX Export
# -----------------------
def export_model_onnx(model, export_path):
    try:

        onnx_file = model.export(format="onnx", dynamic=True)
        export_dir = os.path.dirname(export_path)
        os.makedirs(export_dir, exist_ok=True)
        shutil.copy2(onnx_file, export_path)

        logger.info(f"ℹ️ONNX model copied to: {export_path}")

    except Exception as e:
        logger.error(f"❌Failed to export model to ONNX: {e}", exc_info=True)
        raise

# -----------------------
# Save_Metrics
# -----------------------
def save_metrics_for_pipeline(model_dir, output_dir):
    results_csv = os.path.join(model_dir, "train", "results.csv")

    if not os.path.exists(results_csv):
        logger.error(f"❌ results.csv not found at {results_csv}")
        raise FileNotFoundError(f"results.csv not found")

    df = pd.read_csv(results_csv)
    last_row = df.iloc[-1]  # Last epoch index

    recall = float(last_row.get("metrics/recall(B)", 0))
    map_50 = float(last_row.get("metrics/mAP50(B)", 0))
    #accuracy = float(last_row.get("metrics/class_accuracy", 0))

    metrics = {
        "recall": recall,
        "map_50": map_50
    }

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    logger.info(f"💡 Evaluation metrics - recall: {recall:.4f}, mAP@0.5: {map_50:.4f}")
    logger.info(f"ℹ️ Saved metrics to {metrics_path}: {metrics}")

    return metrics

# -----------------------
# Send_Metrics_to_Cloudwatch
# -----------------------
def send_metrics_to_cloudwatch(recall, map_50):
    try:
        # Create CloudWatchClient
        cloudwatch = boto3.client('cloudwatch', region_name=os.environ.get('AWS_REGION', 'us-east-1')) #eu-central-1

        # Metric namespace and dimension settings
        namespace = 'YOLOTrainingMetrics'
        job_name = os.environ.get('SM_JOB_NAME', 'UnknownJob')

        metrics = [
            {
                'MetricName': 'Recall',
                'Dimensions': [{'Name': 'TrainingJobName', 'Value': job_name}],
                'Value': recall,
                'Unit': 'None'
            },
            {
                'MetricName': 'mAP_0.5',
                'Dimensions': [{'Name': 'TrainingJobName', 'Value': job_name}],
                'Value': map_50,
                'Unit': 'None'
            }
        ]

        # Send metrics to CloudWatch
        cloudwatch.put_metric_data(Namespace=namespace, MetricData=metrics)
        logger.info(f"ℹ️Sent metrics to CloudWatch: recall={recall}, mAP@0.5={map_50}")

    except Exception as e:
        logger.error(f"Failed to send metrics to CloudWatch: {e}", exc_info=True)

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    try:
        logger.info("ℹ️Trainning started.")
        input_data_dir, config_dir, model_dir, output_dir = get_sagemaker_paths()
        logger.info(f"ℹ️Input data directory: {input_data_dir}")
        logger.info(f"ℹ️Config directory: {config_dir}")
        logger.info(f"ℹ️Model directory: {model_dir}")
        # logger.info(f"Config directory: {config_dir}")
        logger.info(f"ℹ️Output directory: {output_dir}")

        # Read Config file
        config_path = os.path.join(config_dir, "config.yaml")
        #config_path = "config.yaml"
        config, filtered = load_config(config_path)
        logger.info(f"ℹ️Loaded config from {config_path}: {config}")

        # Model learning
        logger.info("ℹ️Starting training")
        trained_model = train_model(config, input_data_dir, model_dir)
        logger.info("ℹ️Training finished.")

        # ONNX Export
        logger.info("ℹ️Exporting model to ONNX format")
        export_model_onnx(trained_model, os.path.join(model_dir, "model.onnx"))

        # Save Metrics for condition step
        metrics = save_metrics_for_pipeline(model_dir, output_dir)

        # Send Metrics for Cloudwatch
        send_metrics_to_cloudwatch(metrics.get('recall'), metrics.get('map_50'))

        logger.info("🔵Trainning finished successfully.")

    except Exception as e:
        logger.error(f"❌Script failed: {e}", exc_info=True)
        exit(1)
