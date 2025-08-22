"""
Minimal YOLO evaluation entrypoint for SageMaker.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import yaml




def install_ultralytics():
    """Install ultralytics if not available."""
    try:
        import ultralytics
        print("ultralytics already available")
        return True
    except ImportError:
        print("Installing ultralytics...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--quiet"])
            import ultralytics
            print("ultralytics installed successfully")
            return True
        except Exception as e:
            print(f"Failed to install ultralytics: {e}")
            return False

# Install and import
if not install_ultralytics():
    sys.exit(1)

from ultralytics import YOLO
import torch

def find_config_file():
    """Find config file in various possible locations."""
    possible_paths = [
        "/opt/ml/processing/input_config.yaml/config.yaml",  # Add this line
        "/opt/ml/processing/config.yaml",
        "/opt/ml/processing/input_config.yaml",  # New location from pipeline
        "/opt/ml/processing/config.yaml",
        "/opt/ml/processing/config/config.yaml",
        "/opt/ml/processing/input/config/config.yaml"
    ]
    
    # Debug: List contents of /opt/ml/processing
    print("Debug: Contents of /opt/ml/processing:")
    try:
        for item in os.listdir("/opt/ml/processing"):
            item_path = os.path.join("/opt/ml/processing", item)
            if os.path.isdir(item_path):
                print(f"  📁 {item}/ (directory)")
            else:
                print(f"  📄 {item} (file)")
    except Exception as e:
        print(f"  Error listing directory: {e}")
    
    for path in possible_paths:
        print(f"Checking config path: {path}")
        try:
            if os.path.isfile(path):
                print(f"✓ Found config at: {path}")
                return path
            elif os.path.isdir(path):
                print(f"⚠️ Path exists but is a directory: {path}")
                # List contents of this directory
                try:
                    dir_contents = os.listdir(path)
                    print(f"  Directory contents: {dir_contents}")
                except Exception as e:
                    print(f"  Error listing directory contents: {e}")
            else:
                print(f"✗ Config not found at: {path}")
        except Exception as e:
            print(f"✗ Error checking path {path}: {e}")
    
    print("No config file found, creating default config")
    # Create a default config file
    default_config = {
        'evaluation': {
            's3_test_dataset': '/opt/ml/processing/test_data',
            'trained_model': '/opt/ml/processing/trained_model',
            'yolo_baseline_model': 'yolo11n',
            'instance_type': 'ml.m5.xlarge',
            'instance_count': 1,
            'volume_size': 30,
            'max_runtime': 3600,
            'metrics': {
                'confidence_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_detections': 300
            },
            'output': {
                'save_predictions': True,
                'save_visualizations': False,
                'detailed_metrics': True
            }
        }
    }
    
    # Use a unique filename to avoid conflicts
    default_config_path = "/opt/ml/processing/default_eval_config.yaml"
    try:
        with open(default_config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print(f"✓ Created default config at: {default_config_path}")
        return default_config_path
    except Exception as e:
        print(f"⚠️ Failed to create default config at {default_config_path}: {e}")
        # Try alternative location
        alt_config_path = "/tmp/default_eval_config.yaml"
        try:
            with open(alt_config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            print(f"✓ Created default config at alternative location: {alt_config_path}")
            return alt_config_path
        except Exception as alt_e:
            print(f"✗ Failed to create config at alternative location: {alt_e}")
            raise RuntimeError("Could not create default config file")

def find_test_data():
    """Find test dataset yaml file."""
    test_dirs = [
        "/opt/ml/processing/test_data",
        "/opt/ml/processing/input/test_data"
    ]
    
    for test_dir in test_dirs:
        print(f"Checking test directory: {test_dir}")
        if os.path.exists(test_dir):
            print(f"✓ Directory exists: {test_dir}")
            # Look for data.yaml
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file == "data.yaml":
                        data_yaml = os.path.join(root, file)
                        print(f"Found test data config: {data_yaml}")
                        return data_yaml
        else:
            print(f"✗ Directory does not exist: {test_dir}")
    
    raise FileNotFoundError("Could not find data.yaml in test data")

def find_trained_model():
    """Find and extract trained model file."""
    import tarfile
    
    model_dirs = [
        "/opt/ml/processing/trained_model",
        "/opt/ml/processing/input/trained_model"
    ]
    
    print("Debugging model directory search...")
    for i, model_dir in enumerate(model_dirs):
        print(f"Model dir {i+1}: {model_dir}")
        if os.path.exists(model_dir):
            print(f"✓ Model directory exists: {model_dir}")
            
            # Debug: List all files in the directory
            print("Available files:")
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"  {file_path} ({file_size} bytes)")
            
            # First look for .pt files directly
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith('.pt'):
                        model_path = os.path.join(root, file)
                        print(f"Found trained model: {model_path}")
                        return model_path
            
            # If no .pt files, look for tar.gz files to extract
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    if file.endswith('.tar.gz'):
                        tar_path = os.path.join(root, file)
                        print(f"Found tar.gz file: {tar_path}")
                        
                        # Extract to temporary directory
                        extract_dir = "/tmp/extracted_model"
                        Path(extract_dir).mkdir(parents=True, exist_ok=True)
                        
                        try:
                            with tarfile.open(tar_path, 'r:gz') as tar:
                                tar.extractall(extract_dir)
                                print(f"Extracted to: {extract_dir}")
                            
                            # List extracted files
                            print("Extracted files:")
                            for ext_root, ext_dirs, ext_files in os.walk(extract_dir):
                                for ext_file in ext_files:
                                    ext_file_path = os.path.join(ext_root, ext_file)
                                    print(f"  {ext_file_path}")
                            
                            # Now look for .pt files in extracted directory
                            for ext_root, ext_dirs, ext_files in os.walk(extract_dir):
                                for ext_file in ext_files:
                                    if ext_file.endswith('.pt'):
                                        model_path = os.path.join(ext_root, ext_file)
                                        print(f"Found extracted model: {model_path}")
                                        return model_path
                        
                        except Exception as e:
                            print(f"Failed to extract {tar_path}: {e}")
                            continue
        else:
            print(f"✗ Model directory does not exist: {model_dir}")
    
    print("No model directories found or no model files in directories")
    raise FileNotFoundError("Could not find .pt model file")

def evaluate_model(model, data_yaml, model_name):
    """Run evaluation and return basic metrics."""
    print(f"Evaluating {model_name}...")
    
    try:
        results = model.val(data=data_yaml, conf=0.25, iou=0.45, save=False, plots=False)
        
        metrics = {
            'model_name': model_name,
            'map_50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
            'map_50_95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
            'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
            'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0
        }
        
        print(f"{model_name} - mAP@0.5: {metrics['map_50']:.4f}, mAP@0.5:0.95: {metrics['map_50_95']:.4f}")
        print(f"{model_name} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        return metrics
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return {
            'model_name': model_name, 
            'error': str(e), 
            'map_50': 0.0, 
            'map_50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0
        }


def main():
    print("=" * 50)
    print("MINIMAL YOLO EVALUATION")
    print("=" * 50)
    
    # Setup output directory
    output_dir = "/opt/ml/processing/evaluation"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load config
        config_file = find_config_file()
        if not config_file:
            raise FileNotFoundError("Could not find or create config file")
            
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f).get('evaluation', {})
        
        print(f"✓ Config loaded successfully from: {config_file}")
        print(f"Config contents: {config}")
        
        # Find test data
        print("Searching for test data...")
        data_yaml = find_test_data()
        print(f"✓ Test data found at: {data_yaml}")
        
        # Find trained model
        print("Searching for trained model...")
        trained_model_path = find_trained_model()
        print(f"✓ Trained model found at: {trained_model_path}")
        
        # Load baseline model
        baseline_name = config.get('yolo_baseline_model', 'yolo11n')
        print(f"Loading baseline model: {baseline_name}")
        try:
            baseline_model = YOLO(f"{baseline_name}.pt")
            print(f"✓ Baseline model loaded successfully: {baseline_name}")
        except Exception as e:
            print(f"⚠️ Failed to load baseline model {baseline_name}: {e}")
            print("Trying to download from Ultralytics...")
            try:
                baseline_model = YOLO(baseline_name)  # This will download the model
                print(f"✓ Baseline model downloaded successfully: {baseline_name}")
            except Exception as download_e:
                print(f"✗ Failed to download baseline model: {download_e}")
                raise RuntimeError(f"Could not load or download baseline model {baseline_name}")
        
        # Load trained model
        print(f"Loading trained model from: {trained_model_path}")
        try:
            trained_model = YOLO(trained_model_path)
            print(f"✓ Trained model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load trained model: {e}")
            raise RuntimeError(f"Could not load trained model from {trained_model_path}")
        
        # Run evaluations
        baseline_metrics = evaluate_model(baseline_model, data_yaml, "baseline")
        trained_metrics = evaluate_model(trained_model, data_yaml, "trained")
        
        # Calculate improvements
        map_improvement = trained_metrics['map_50_95'] - baseline_metrics['map_50_95']
        map_improvement_pct = (map_improvement / baseline_metrics['map_50_95'] * 100) if baseline_metrics['map_50_95'] > 0 else 0

        precision_improvement = trained_metrics['precision'] - baseline_metrics['precision']
        precision_improvement_pct = (precision_improvement / baseline_metrics['precision'] * 100) if baseline_metrics['precision'] > 0 else 0

        recall_improvement = trained_metrics['recall'] - baseline_metrics['recall'] 
        recall_improvement_pct = (recall_improvement / baseline_metrics['recall'] * 100) if baseline_metrics['recall'] > 0 else 0

        results = {
            'baseline': baseline_metrics,
            'trained': trained_metrics,
            'improvements': {
                'map_50_95': {'absolute': map_improvement, 'percentage': map_improvement_pct},
                'precision': {'absolute': precision_improvement, 'percentage': precision_improvement_pct},
                'recall': {'absolute': recall_improvement, 'percentage': recall_improvement_pct}
            },
            'status': 'success'
        }

        # Save results
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 50)
        print("EVALUATION COMPLETED")
        print("=" * 50)
        print(f"Baseline mAP@0.5:0.95: {baseline_metrics['map_50_95']:.4f}")
        print(f"Trained mAP@0.5:0.95: {trained_metrics['map_50_95']:.4f}")
        print(f"Improvement: {map_improvement_pct:+.1f}%")
        print(f"Precision: {precision_improvement_pct:+.1f}%")
        print(f"Recall: {recall_improvement_pct:+.1f}%")
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        
        # Save error
        error_results = {'status': 'failed', 'error': str(e)}
        error_file = os.path.join(output_dir, "evaluation_error.json")
        with open(error_file, 'w') as f:
            json.dump(error_results, f, indent=2)
        
        raise

if __name__ == "__main__":
    main()