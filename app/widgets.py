"""Custom widgets for the configuration editor."""
from textual.widgets import Input, Button, Label
from textual.containers import Container, Vertical, Horizontal
from typing import Dict, Any, List, Tuple


class ConfigSection:
    """Defines a configuration section with its fields and types."""
    
    def __init__(self, name: str, fields: List[Tuple[str, type, str]], nested_sections: Dict[str, 'ConfigSection'] = None):
        self.name = name
        self.fields = fields  # (field_name, type, placeholder)
        self.nested_sections = nested_sections or {}


class ConfigInputPanel:
    """Creates input panels for different configuration sections."""
    
    @staticmethod
    def create_inputs_for_section(section: ConfigSection, config_data: Dict[str, Any]) -> Vertical:
        """Create input widgets for a configuration section."""
        inputs = []
        
        # Add section label
        inputs.append(Label(section.name, classes="section"))
        
        # Add inputs for direct fields
        for field_name, field_type, placeholder in section.fields:
            value = config_data.get(field_name, "")
            if value is not None:
                value = str(value)
            else:
                value = ""
            
            inputs.append(Input(
                value=value,
                placeholder=placeholder,
                id=f"{section.name.lower().replace(' ', '_')}_{field_name}"
            ))
        
        # Add nested sections
        for nested_name, nested_section in section.nested_sections.items():
            nested_data = config_data.get(nested_name, {})
            nested_inputs = ConfigInputPanel.create_inputs_for_section(nested_section, nested_data)
            inputs.append(nested_inputs)
        
        # Add save and exit buttons
        inputs.append(
            Horizontal(
                Button("Save Config", variant="success", id="save"),
                Button("Exit", variant="error", id="exit")
            )
        )
        
        return Vertical(*inputs)


# Define all configuration sections based on the config.yaml structure
CONFIG_SECTIONS = {
    "AWS Configuration": ConfigSection(
        "AWS Configuration",
        [
            ("bucket", str, "S3 Bucket Name"),
            ("prefix", str, "S3 Prefix"),
            ("region", str, "AWS Region"),
            ("role_arn", str, "IAM Role ARN"),
        ]
    ),
    
    "Training Configuration": ConfigSection(
        "Training Configuration", 
        [
            ("s3_dataset_prefix", str, "S3 Dataset Prefix"),
            ("s3_train_prefix", str, "S3 Train Prefix"),
            ("s3_val_prefix", str, "S3 Validation Prefix"),
            ("models_dir", str, "Models Directory"),
            ("model_dir", str, "Model Directory"),
            ("output_dir", str, "Output Directory"),
            ("model_name", str, "Model Name"),
            ("instance_type", str, "Instance Type"),
            ("instance_count", int, "Instance Count"),
            ("volume_size", int, "Volume Size"),
            ("max_run", int, "Max Run Time (seconds)"),
        ],
        nested_sections={
            "hyperparams": ConfigSection(
                "Hyperparameters",
                [
                    ("imgsz", int, "Image Size"),
                    ("epochs", int, "Epochs"),
                    ("batch", int, "Batch Size"),
                    ("lr0", float, "Initial Learning Rate"),
                    ("optimizer", str, "Optimizer"),
                ]
            )
        }
    ),
    
    "Evaluation Configuration": ConfigSection(
        "Evaluation Configuration",
        [
            ("evaluation_step_name", str, "Evaluation Step Name"),
            ("instance_count", int, "Instance Count"),
            ("instance_type", str, "Instance Type"),
            ("max_runtime", int, "Max Runtime (seconds)"),
            ("name", str, "Pipeline Name"),
            ("s3_test_dataset", str, "S3 Test Dataset"),
            ("trained_model_job", str, "Trained Model Job"),
            ("volume_size", int, "Volume Size"),
            ("yolo_baseline_model", str, "YOLO Baseline Model"),
        ],
        nested_sections={
            "metrics": ConfigSection(
                "Metrics Configuration",
                [
                    ("confidence_threshold", float, "Confidence Threshold"),
                    ("iou_threshold", float, "IoU Threshold"),
                    ("max_detections", int, "Max Detections"),
                ]
            ),
            "output": ConfigSection(
                "Output Configuration",
                [
                    ("save_predictions", bool, "Save Predictions"),
                    ("save_visualizations", bool, "Save Visualizations"),
                ]
            )
        }
    ),
    
    "Inference Configuration": ConfigSection(
        "Inference Configuration",
        [
            ("batch_size", int, "Batch Size"),
            ("confidence_threshold", float, "Confidence Threshold"),
            ("iou_threshold", float, "IoU Threshold"),
            ("job_name", str, "Job Name"),
            ("max_image_size", int, "Max Image Size"),
            ("metric_key", str, "Metric Key"),
            ("model_data", str, "Model Data S3 Path"),
            ("output_dir", str, "Output Directory"),
            ("s3_inference_dataset", str, "S3 Inference Dataset"),
        ]
    ),
    
    "Pipeline Configuration": ConfigSection(
        "Pipeline Configuration",
        [
            ("dry_run", bool, "Dry Run"),
            ("enable_caching", bool, "Enable Caching"),
            ("execution_name", str, "Execution Name"),
            ("model_package_group_name", str, "Model Package Group Name"),
            ("name", str, "Pipeline Name"),
            ("registration_step_name", str, "Registration Step Name"),
            ("training_step_name", str, "Training Step Name"),
        ]
    ),
    
    "Tuning Configuration": ConfigSection(
        "Tuning Configuration",
        [
            ("enabled", bool, "Enable Hyperparameter Tuning"),
            ("max_jobs", int, "Max Jobs"),
            ("max_parallel_jobs", int, "Max Parallel Jobs"),
            ("objective_metric", str, "Objective Metric"),
            ("objective_type", str, "Objective Type"),
            ("update_config", bool, "Update Config"),
        ]
    ),
}
