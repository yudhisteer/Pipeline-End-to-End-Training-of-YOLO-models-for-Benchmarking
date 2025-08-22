# """
# Minimal YOLO evaluation entrypoint for SageMaker.
# """

# import os
# import sys
# import subprocess
# import json
# from pathlib import Path
# import yaml




# def install_ultralytics():
#     """Install ultralytics if not available."""
#     try:
#         import ultralytics
#         print("ultralytics already available")
#         return True
#     except ImportError:
#         print("Installing ultralytics...")
#         try:
#             subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--quiet"])
#             import ultralytics
#             print("ultralytics installed successfully")
#             return True
#         except Exception as e:
#             print(f"Failed to install ultralytics: {e}")
#             return False

# # Install and import
# if not install_ultralytics():
#     sys.exit(1)

# from ultralytics import YOLO
# import torch

# def find_config_file():
#     """Find config file in various possible locations."""
#     possible_paths = [
#         "/opt/ml/processing/input_config.yaml/config.yaml",  # Add this line
#         "/opt/ml/processing/config.yaml",
#         "/opt/ml/processing/input_config.yaml",  # New location from pipeline
#         "/opt/ml/processing/config.yaml",
#         "/opt/ml/processing/config/config.yaml",
#         "/opt/ml/processing/input/config/config.yaml"
#     ]
    
#     # Debug: List contents of /opt/ml/processing
#     print("Debug: Contents of /opt/ml/processing:")
#     try:
#         for item in os.listdir("/opt/ml/processing"):
#             item_path = os.path.join("/opt/ml/processing", item)
#             if os.path.isdir(item_path):
#                 print(f"  📁 {item}/ (directory)")
#             else:
#                 print(f"  📄 {item} (file)")
#     except Exception as e:
#         print(f"  Error listing directory: {e}")
    
#     for path in possible_paths:
#         print(f"Checking config path: {path}")
#         try:
#             if os.path.isfile(path):
#                 print(f"✓ Found config at: {path}")
#                 return path
#             elif os.path.isdir(path):
#                 print(f"⚠️ Path exists but is a directory: {path}")
#                 # List contents of this directory
#                 try:
#                     dir_contents = os.listdir(path)
#                     print(f"  Directory contents: {dir_contents}")
#                 except Exception as e:
#                     print(f"  Error listing directory contents: {e}")
#             else:
#                 print(f"✗ Config not found at: {path}")
#         except Exception as e:
#             print(f"✗ Error checking path {path}: {e}")
    
#     print("No config file found, creating default config")
#     # Create a default config file
#     default_config = {
#         'evaluation': {
#             's3_test_dataset': '/opt/ml/processing/test_data',
#             'trained_model': '/opt/ml/processing/trained_model',
#             'yolo_baseline_model': 'yolo11n',
#             'instance_type': 'ml.m5.xlarge',
#             'instance_count': 1,
#             'volume_size': 30,
#             'max_runtime': 3600,
#             'metrics': {
#                 'confidence_threshold': 0.25,
#                 'iou_threshold': 0.45,
#                 'max_detections': 300
#             },
#             'output': {
#                 'save_predictions': True,
#                 'save_visualizations': False,
#                 'detailed_metrics': True
#             }
#         }
#     }
    
#     # Use a unique filename to avoid conflicts
#     default_config_path = "/opt/ml/processing/default_eval_config.yaml"
#     try:
#         with open(default_config_path, 'w') as f:
#             yaml.dump(default_config, f, default_flow_style=False)
#         print(f"✓ Created default config at: {default_config_path}")
#         return default_config_path
#     except Exception as e:
#         print(f"⚠️ Failed to create default config at {default_config_path}: {e}")
#         # Try alternative location
#         alt_config_path = "/tmp/default_eval_config.yaml"
#         try:
#             with open(alt_config_path, 'w') as f:
#                 yaml.dump(default_config, f, default_flow_style=False)
#             print(f"✓ Created default config at alternative location: {alt_config_path}")
#             return alt_config_path
#         except Exception as alt_e:
#             print(f"✗ Failed to create config at alternative location: {alt_e}")
#             raise RuntimeError("Could not create default config file")

# def find_test_data():
#     """Find test dataset yaml file."""
#     test_dirs = [
#         "/opt/ml/processing/test_data",
#         "/opt/ml/processing/input/test_data"
#     ]
    
#     for test_dir in test_dirs:
#         print(f"Checking test directory: {test_dir}")
#         if os.path.exists(test_dir):
#             print(f"✓ Directory exists: {test_dir}")
#             # Look for data.yaml
#             for root, dirs, files in os.walk(test_dir):
#                 for file in files:
#                     if file == "data.yaml":
#                         data_yaml = os.path.join(root, file)
#                         print(f"Found test data config: {data_yaml}")
#                         return data_yaml
#         else:
#             print(f"✗ Directory does not exist: {test_dir}")
    
#     raise FileNotFoundError("Could not find data.yaml in test data")

# def find_trained_model():
#     """Find and extract trained model file."""
#     import tarfile
    
#     model_dirs = [
#         "/opt/ml/processing/trained_model",
#         "/opt/ml/processing/input/trained_model"
#     ]
    
#     print("Debugging model directory search...")
#     for i, model_dir in enumerate(model_dirs):
#         print(f"Model dir {i+1}: {model_dir}")
#         if os.path.exists(model_dir):
#             print(f"✓ Model directory exists: {model_dir}")
            
#             # Debug: List all files in the directory
#             print("Available files:")
#             for root, dirs, files in os.walk(model_dir):
#                 for file in files:
#                     file_path = os.path.join(root, file)
#                     file_size = os.path.getsize(file_path)
#                     print(f"  {file_path} ({file_size} bytes)")
            
#             # First look for .pt files directly
#             for root, dirs, files in os.walk(model_dir):
#                 for file in files:
#                     if file.endswith('.pt'):
#                         model_path = os.path.join(root, file)
#                         print(f"Found trained model: {model_path}")
#                         return model_path
            
#             # If no .pt files, look for tar.gz files to extract
#             for root, dirs, files in os.walk(model_dir):
#                 for file in files:
#                     if file.endswith('.tar.gz'):
#                         tar_path = os.path.join(root, file)
#                         print(f"Found tar.gz file: {tar_path}")
                        
#                         # Extract to temporary directory
#                         extract_dir = "/tmp/extracted_model"
#                         Path(extract_dir).mkdir(parents=True, exist_ok=True)
                        
#                         try:
#                             with tarfile.open(tar_path, 'r:gz') as tar:
#                                 tar.extractall(extract_dir)
#                                 print(f"Extracted to: {extract_dir}")
                            
#                             # List extracted files
#                             print("Extracted files:")
#                             for ext_root, ext_dirs, ext_files in os.walk(extract_dir):
#                                 for ext_file in ext_files:
#                                     ext_file_path = os.path.join(ext_root, ext_file)
#                                     print(f"  {ext_file_path}")
                            
#                             # Now look for .pt files in extracted directory
#                             for ext_root, ext_dirs, ext_files in os.walk(extract_dir):
#                                 for ext_file in ext_files:
#                                     if ext_file.endswith('.pt'):
#                                         model_path = os.path.join(ext_root, ext_file)
#                                         print(f"Found extracted model: {model_path}")
#                                         return model_path
                        
#                         except Exception as e:
#                             print(f"Failed to extract {tar_path}: {e}")
#                             continue
#         else:
#             print(f"✗ Model directory does not exist: {model_dir}")
    
#     print("No model directories found or no model files in directories")
#     raise FileNotFoundError("Could not find .pt model file")

# def evaluate_model(model, data_yaml, model_name):
#     """Run evaluation and return basic metrics."""
#     print(f"Evaluating {model_name}...")
    
#     try:
#         results = model.val(data=data_yaml, conf=0.25, iou=0.45, save=False, plots=False)
        
#         metrics = {
#             'model_name': model_name,
#             'map_50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
#             'map_50_95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
#             'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
#             'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0
#         }
        
#         print(f"{model_name} - mAP@0.5: {metrics['map_50']:.4f}, mAP@0.5:0.95: {metrics['map_50_95']:.4f}")
#         print(f"{model_name} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
#         return metrics
        
#     except Exception as e:
#         print(f"Error evaluating {model_name}: {e}")
#         return {
#             'model_name': model_name, 
#             'error': str(e), 
#             'map_50': 0.0, 
#             'map_50_95': 0.0,
#             'precision': 0.0,
#             'recall': 0.0
#         }


# def main():
#     print("=" * 50)
#     print("MINIMAL YOLO EVALUATION")
#     print("=" * 50)
    
#     # Setup output directory
#     output_dir = "/opt/ml/processing/evaluation"
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
    
#     try:
#         # Load config
#         config_file = find_config_file()
#         if not config_file:
#             raise FileNotFoundError("Could not find or create config file")
            
#         with open(config_file, 'r') as f:
#             config = yaml.safe_load(f).get('evaluation', {})
        
#         print(f"✓ Config loaded successfully from: {config_file}")
#         print(f"Config contents: {config}")
        
#         # Find test data
#         print("Searching for test data...")
#         data_yaml = find_test_data()
#         print(f"✓ Test data found at: {data_yaml}")
        
#         # Find trained model
#         print("Searching for trained model...")
#         trained_model_path = find_trained_model()
#         print(f"✓ Trained model found at: {trained_model_path}")
        
#         # Load baseline model
#         baseline_name = config.get('yolo_baseline_model', 'yolo11n')
#         print(f"Loading baseline model: {baseline_name}")
#         try:
#             baseline_model = YOLO(f"{baseline_name}.pt")
#             print(f"✓ Baseline model loaded successfully: {baseline_name}")
#         except Exception as e:
#             print(f"⚠️ Failed to load baseline model {baseline_name}: {e}")
#             print("Trying to download from Ultralytics...")
#             try:
#                 baseline_model = YOLO(baseline_name)  # This will download the model
#                 print(f"✓ Baseline model downloaded successfully: {baseline_name}")
#             except Exception as download_e:
#                 print(f"✗ Failed to download baseline model: {download_e}")
#                 raise RuntimeError(f"Could not load or download baseline model {baseline_name}")
        
#         # Load trained model
#         print(f"Loading trained model from: {trained_model_path}")
#         try:
#             trained_model = YOLO(trained_model_path)
#             print(f"✓ Trained model loaded successfully")
#         except Exception as e:
#             print(f"✗ Failed to load trained model: {e}")
#             raise RuntimeError(f"Could not load trained model from {trained_model_path}")
        
#         # Run evaluations
#         baseline_metrics = evaluate_model(baseline_model, data_yaml, "baseline")
#         trained_metrics = evaluate_model(trained_model, data_yaml, "trained")
        
#         # Calculate improvements
#         map_improvement = trained_metrics['map_50_95'] - baseline_metrics['map_50_95']
#         map_improvement_pct = (map_improvement / baseline_metrics['map_50_95'] * 100) if baseline_metrics['map_50_95'] > 0 else 0

#         precision_improvement = trained_metrics['precision'] - baseline_metrics['precision']
#         precision_improvement_pct = (precision_improvement / baseline_metrics['precision'] * 100) if baseline_metrics['precision'] > 0 else 0

#         recall_improvement = trained_metrics['recall'] - baseline_metrics['recall'] 
#         recall_improvement_pct = (recall_improvement / baseline_metrics['recall'] * 100) if baseline_metrics['recall'] > 0 else 0

#         results = {
#             'baseline': baseline_metrics,
#             'trained': trained_metrics,
#             'improvements': {
#                 'map_50_95': {'absolute': map_improvement, 'percentage': map_improvement_pct},
#                 'precision': {'absolute': precision_improvement, 'percentage': precision_improvement_pct},
#                 'recall': {'absolute': recall_improvement, 'percentage': recall_improvement_pct}
#             },
#             'status': 'success'
#         }

#         # Save results
#         results_file = os.path.join(output_dir, "evaluation_results.json")
#         with open(results_file, 'w') as f:
#             json.dump(results, f, indent=2)
        
#         print("\n" + "=" * 50)
#         print("EVALUATION COMPLETED")
#         print("=" * 50)
#         print(f"Baseline mAP@0.5:0.95: {baseline_metrics['map_50_95']:.4f}")
#         print(f"Trained mAP@0.5:0.95: {trained_metrics['map_50_95']:.4f}")
#         print(f"Improvement: {map_improvement_pct:+.1f}%")
#         print(f"Precision: {precision_improvement_pct:+.1f}%")
#         print(f"Recall: {recall_improvement_pct:+.1f}%")
#         print(f"Results saved to: {results_file}")
        
#     except Exception as e:
#         print(f"Evaluation failed: {e}")
        
#         # Save error
#         error_results = {'status': 'failed', 'error': str(e)}
#         error_file = os.path.join(output_dir, "evaluation_error.json")
#         with open(error_file, 'w') as f:
#             json.dump(error_results, f, indent=2)
        
#         raise

# if __name__ == "__main__":
#     main()

"""
YOLO Model Evaluation for SageMaker Processing Jobs
"""

import os
import sys
import subprocess
import json
import yaml
import tarfile
from pathlib import Path


def install_ultralytics():
    """Install ultralytics if not available."""
    try:
        import ultralytics
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "matplotlib", "--quiet"])
            return True
        except Exception:
            return False


# Install dependencies
if not install_ultralytics():
    sys.exit(1)

from ultralytics import YOLO


def load_config():
    """Load evaluation configuration."""
    config_path = "/opt/ml/processing/input_config.yaml/config.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f).get('evaluation', {})


def extract_trained_model():
    """Extract trained model from tar.gz."""
    tar_path = "/opt/ml/processing/trained_model/model.tar.gz"
    extract_dir = "/tmp/extracted_model"
    
    Path(extract_dir).mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    
    # Return best.onnx
    best_onnx = "/tmp/extracted_model/train/weights/best.onnx"
    
    if os.path.exists(best_onnx):
        return best_onnx
    else:
        raise FileNotFoundError("Could not find best.onnx model")


def evaluate_model(model, data_yaml, model_name, config):
    """Evaluate model and return metrics."""
    metrics_config = config.get('metrics', {})
    
    results = model.val(
        data=data_yaml,
        conf=metrics_config.get('confidence_threshold', 0.25),
        iou=metrics_config.get('iou_threshold', 0.45),
        save=False,
        plots=False
    )
    
    return {
        'model_name': model_name,
        'map_50': float(results.box.map50) if hasattr(results.box, 'map50') else 0.0,
        'map_50_95': float(results.box.map) if hasattr(results.box, 'map') else 0.0,
        'precision': float(results.box.mp) if hasattr(results.box, 'mp') else 0.0,
        'recall': float(results.box.mr) if hasattr(results.box, 'mr') else 0.0
    }


def calculate_improvements(baseline_metrics, trained_metrics):
    """Calculate improvement metrics."""
    improvements = {}
    
    for metric in ['map_50_95', 'precision', 'recall']:
        baseline_val = baseline_metrics.get(metric, 0)
        trained_val = trained_metrics.get(metric, 0)
        
        absolute_change = trained_val - baseline_val
        percentage_change = (absolute_change / baseline_val * 100) if baseline_val > 0 else 0
        
        improvements[metric] = {
            'absolute': absolute_change,
            'percentage': percentage_change
        }
    
    return improvements


def generate_evaluation_charts(results, output_dir):
   """Generate comparison charts from evaluation results."""
   try:
       import matplotlib.pyplot as plt
       import numpy as np
   except ImportError:
       print("matplotlib not available, skipping chart generation")
       return
   
   baseline = results['baseline']
   trained = results['trained']
   improvements = results['improvements']
   
   # Set up the plotting style
   plt.style.use('default')
   fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
   fig.suptitle('YOLO Model Evaluation Comparison', fontsize=16, fontweight='bold')
   
   # 1. Metrics Comparison Bar Chart
   metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']
   baseline_values = [baseline['map_50'], baseline['map_50_95'], baseline['precision'], baseline['recall']]
   trained_values = [trained['map_50'], trained['map_50_95'], trained['precision'], trained['recall']]
   
   x = np.arange(len(metrics))
   width = 0.35
   
   ax1.bar(x - width/2, baseline_values, width, label='Baseline', color='#2E8B57', alpha=0.8)
   ax1.bar(x + width/2, trained_values, width, label='Trained', color='#FF6347', alpha=0.8)
   ax1.set_xlabel('Metrics')
   ax1.set_ylabel('Score')
   ax1.set_title('Performance Metrics Comparison')
   ax1.set_xticks(x)
   ax1.set_xticklabels(metrics, rotation=45)
   ax1.legend()
   ax1.grid(True, alpha=0.3)
   ax1.set_ylim(0, 1)
   
   # 2. Improvement Percentage Chart
   improvement_metrics = list(improvements.keys())
   improvement_values = [improvements[metric]['percentage'] for metric in improvement_metrics]
   colors = ['green' if val > 0 else 'red' for val in improvement_values]
   
   bars = ax2.bar(improvement_metrics, improvement_values, color=colors, alpha=0.7)
   ax2.set_xlabel('Metrics')
   ax2.set_ylabel('Improvement (%)')
   ax2.set_title('Performance Changes (%)')
   ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
   ax2.grid(True, alpha=0.3)
   
   # Add value labels on bars
   for bar, value in zip(bars, improvement_values):
       height = bar.get_height()
       ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
               f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
   
   # 3. Radar Chart for Metrics
   radar_metrics = ['mAP@0.5:0.95', 'Precision', 'Recall']
   radar_baseline = [baseline['map_50_95'], baseline['precision'], baseline['recall']]
   radar_trained = [trained['map_50_95'], trained['precision'], trained['recall']]
   
   angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
   angles = np.concatenate((angles, [angles[0]]))
   
   radar_baseline_closed = radar_baseline + [radar_baseline[0]]
   radar_trained_closed = radar_trained + [radar_trained[0]]
   
   ax3.plot(angles, radar_baseline_closed, 'o-', linewidth=2, label='Baseline', color='#2E8B57')
   ax3.fill(angles, radar_baseline_closed, alpha=0.25, color='#2E8B57')
   ax3.plot(angles, radar_trained_closed, 'o-', linewidth=2, label='Trained', color='#FF6347')
   ax3.fill(angles, radar_trained_closed, alpha=0.25, color='#FF6347')
   
   ax3.set_xticks(angles[:-1])
   ax3.set_xticklabels(radar_metrics)
   ax3.set_ylim(0, 1)
   ax3.set_title('Performance Radar Chart')
   ax3.legend()
   ax3.grid(True)
   
   # 4. Summary Statistics
   ax4.axis('off')
   summary_text = f"""Model Performance Summary

Baseline Model:
- mAP@0.5:0.95: {baseline['map_50_95']:.3f}
- Precision: {baseline['precision']:.3f}
- Recall: {baseline['recall']:.3f}

Trained Model:
- mAP@0.5:0.95: {trained['map_50_95']:.3f}
- Precision: {trained['precision']:.3f}
- Recall: {trained['recall']:.3f}

Overall Change:
- mAP Improvement: {improvements['map_50_95']['percentage']:+.1f}%
- Precision Change: {improvements['precision']['percentage']:+.1f}%
- Recall Change: {improvements['recall']['percentage']:+.1f}%"""
   
   ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
   
   plt.tight_layout()
   
   # Save the chart
   chart_file = os.path.join(output_dir, "evaluation_comparison_charts.png")
   plt.savefig(chart_file, dpi=300, bbox_inches='tight')
   plt.close()
   
   print(f"Evaluation charts saved to: {chart_file}")

   _generate_metric_trend_chart(results, output_dir)


def _generate_metric_trend_chart(results, output_dir):
   """Generate a simple metric comparison chart."""
   try:
       import matplotlib.pyplot as plt
       import numpy as np
   except ImportError:
       return
   
   baseline = results['baseline']
   trained = results['trained']
   
   fig, ax = plt.subplots(1, 1, figsize=(10, 6))
   
   metrics = ['mAP@0.5:0.95', 'Precision', 'Recall']
   baseline_vals = [baseline['map_50_95'], baseline['precision'], baseline['recall']]
   trained_vals = [trained['map_50_95'], trained['precision'], trained['recall']]
   
   colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
   
   # Create separate lines for each metric
   for i, (metric, baseline_val, trained_val, color) in enumerate(zip(metrics, baseline_vals, trained_vals, colors)):
       ax.plot([0, 1], [baseline_val, trained_val], 'o-', linewidth=3, markersize=8, 
               color=color, label=metric)
       
       # Add percentage change labels
       change = ((trained_val - baseline_val) / baseline_val) * 100
       mid_x, mid_y = 0.5, (baseline_val + trained_val) / 2
       ax.annotate(f'{change:+.1f}%', xy=(mid_x, mid_y), xytext=(10, 10), 
                  textcoords='offset points', ha='left',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3))
   
   ax.set_xlim(-0.1, 1.1)
   ax.set_xticks([0, 1])
   ax.set_xticklabels(['Baseline', 'Trained'], fontsize=12)
   ax.set_ylabel('Score', fontsize=12)
   ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
   ax.grid(True, alpha=0.3)
   ax.set_ylim(0, 1)
   ax.legend(loc='upper right')
   
   plt.tight_layout()
   trend_file = os.path.join(output_dir, "metric_trends.png")
   plt.savefig(trend_file, dpi=300, bbox_inches='tight')
   plt.close()
   
   print(f"Metric trends chart saved to: {trend_file}")



def main():
    """Main evaluation function."""
    print("YOLO Model Evaluation Starting...")
    
    # Setup
    output_dir = "/opt/ml/processing/evaluation"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load configuration
        config = load_config()
        
        # Use direct paths
        data_yaml = "/opt/ml/processing/test_data/data.yaml"
        trained_model_path = extract_trained_model()
        
        # Load models
        baseline_name = config.get('yolo_baseline_model', 'yolo11n')
        baseline_model = YOLO(f"{baseline_name}.pt")
        trained_model = YOLO(trained_model_path)
        
        # Run evaluations
        baseline_metrics = evaluate_model(baseline_model, data_yaml, "baseline", config)
        trained_metrics = evaluate_model(trained_model, data_yaml, "trained", config)
        
        # Calculate improvements
        improvements = calculate_improvements(baseline_metrics, trained_metrics)
        
        # Prepare results
        results = {
            'baseline': baseline_metrics,
            'trained': trained_metrics,
            'improvements': improvements,
            'status': 'success'
        }
        
        # Save results
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Generate evaluation charts
        generate_evaluation_charts(results, output_dir)
        
        # Print summary
        print(f"Baseline mAP@0.5:0.95: {baseline_metrics['map_50_95']:.4f}")
        print(f"Trained mAP@0.5:0.95: {trained_metrics['map_50_95']:.4f}")
        print(f"Improvement: {improvements['map_50_95']['percentage']:+.1f}%")
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        # Save error
        error_results = {'status': 'failed', 'error': str(e)}
        error_file = os.path.join(output_dir, "evaluation_error.json")
        with open(error_file, 'w') as f:
            json.dump(error_results, f, indent=2)
        raise


if __name__ == "__main__":
    main()