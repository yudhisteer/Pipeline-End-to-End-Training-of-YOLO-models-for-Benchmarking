"""
Minimal YOLO evaluation entrypoint for SageMaker.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

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
        "/opt/ml/processing/config.yaml",
        "/opt/ml/processing/config/config.yaml",
        "/opt/ml/processing/input/config/config.yaml"
    ]
    
    for path in possible_paths:
        if os.path.isfile(path):
            print(f"Found config at: {path}")
            return path
    
    print("No config file found, using defaults")
    return None

def get_default_config():
    """Return minimal default config."""
    return {
        'yolo_baseline_model': 'yolo11n',
        'confidence_threshold': 0.25,
        'iou_threshold': 0.45
    }

def find_test_data():
    """Find test dataset yaml file."""
    test_dirs = [
        "/opt/ml/processing/test_data",
        "/opt/ml/processing/input/test_data"
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            # Look for data.yaml
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file == "data.yaml":
                        data_yaml = os.path.join(root, file)
                        print(f"Found test data config: {data_yaml}")
                        return data_yaml
    
    raise FileNotFoundError("Could not find data.yaml in test data")

def find_trained_model():
    """Find and extract trained model file."""
    import tarfile
    
    model_dirs = [
        "/opt/ml/processing/trained_model",
        "/opt/ml/processing/input/trained_model"
    ]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"Searching for model in: {model_dir}")
            
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
            'map_50_95': float(results.box.map) if hasattr(results.box, 'map') else 0.0
        }
        
        print(f"{model_name} - mAP@0.5: {metrics['map_50']:.4f}, mAP@0.5:0.95: {metrics['map_50_95']:.4f}")
        return metrics
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return {'model_name': model_name, 'error': str(e), 'map_50': 0.0, 'map_50_95': 0.0}

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
        if config_file:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f).get('evaluation', {})
        else:
            config = get_default_config()
        
        # Find test data
        data_yaml = find_test_data()
        
        # Load baseline model
        baseline_name = config.get('yolo_baseline_model', 'yolo11n')
        print(f"Loading baseline model: {baseline_name}")
        baseline_model = YOLO(f"{baseline_name}.pt")
        
        # Load trained model
        trained_model_path = find_trained_model()
        print(f"Loading trained model from: {trained_model_path}")
        trained_model = YOLO(trained_model_path)
        
        # Run evaluations
        baseline_metrics = evaluate_model(baseline_model, data_yaml, "baseline")
        trained_metrics = evaluate_model(trained_model, data_yaml, "trained")
        
        # Simple comparison
        improvement = trained_metrics['map_50_95'] - baseline_metrics['map_50_95']
        improvement_pct = (improvement / baseline_metrics['map_50_95'] * 100) if baseline_metrics['map_50_95'] > 0 else 0
        
        results = {
            'baseline': baseline_metrics,
            'trained': trained_metrics,
            'improvement_absolute': improvement,
            'improvement_percentage': improvement_pct,
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
        print(f"Improvement: {improvement_pct:+.1f}%")
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