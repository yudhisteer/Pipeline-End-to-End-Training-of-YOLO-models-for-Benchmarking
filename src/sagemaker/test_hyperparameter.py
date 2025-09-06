import yaml
import re
from datetime import datetime
from typing import Dict, Any
from rich import print


from sagemaker.tuner import HyperparameterTuner
from sagemaker.tuner import ContinuousParameter, CategoricalParameter, IntegerParameter

from entrypoint_trainer import YOLOSageMakerTrainer
from utils.utils_config import load_config


# some dummy hyperparams
tuning_config = {
    "tuning": {
    "enabled": True,                    # Enable for hyperparameter tuning
    "update_config": True,                  # Update config.yaml with best hyperparameters after tuning
    "max_jobs": 6,                      # Small number for POC
    "max_parallel_jobs": 2,             # Limited parallel jobs
    "objective_metric": "mAP_0.5",        # Primary YOLO metric
    "objective_type": "Maximize",         # We want to maximize mAP
    "strategy": "Hyperband",

    "hyperparameter_ranges": {
        "lr0": {
        "type": "continuous",    
        "scaling_type": "Logarithmic",
        "min": 0.1,
        "max": 0.2,
        },
        "batch": {
        "type": "categorical",
        "values": [2, 4],
        },
        "optimizer": {  
        "type": "categorical",
        "values": ["SGD", "Adam"],
        }
        }
    }
}


def get_hyperparameter_ranges_from_config(tuning_config: dict) -> Dict:
    """
    Create SageMaker hyperparameter ranges from tuning configuration.
    
    Args:
        tuning_config: Tuning configuration dictionary from config.yaml
        
    Returns:
        Dictionary of SageMaker parameter objects
    """
    ranges = {}
    hyperparameter_ranges = tuning_config.get('tuning', {}).get('hyperparameter_ranges', {})
    
    for param_name, param_config in hyperparameter_ranges.items():
        param_type = param_config.get('type')

        if param_type == 'continuous':
            ranges[param_name] = ContinuousParameter(
                param_config['min'], 
                param_config['max'],
                param_config['scaling_type']
            )
        elif param_type == 'categorical':
            ranges[param_name] = CategoricalParameter(param_config['values'])
        elif param_type == 'integer':
            ranges[param_name] = IntegerParameter(
                param_config['min'], 
                param_config['max'],
                param_config['scaling_type']
            )
        else:
            print(f"Warning: Unknown parameter type '{param_type}' for {param_name}")
    
    # print("Hyperparameter ranges for tuning (from config):")
    # for param, range_def in ranges.items():
    #     if isinstance(range_def, ContinuousParameter):
    #         print(f"  {param}: continuous [{range_def.min_value} - {range_def.max_value} ] ({range_def.scaling_type})")
    #     elif isinstance(range_def, CategoricalParameter):
    #         print(f"  {param}: categorical {range_def.values}")
    #     elif isinstance(range_def, IntegerParameter):
    #         print(f"  {param}: integer [{range_def.min_value} - {range_def.max_value}] ({range_def.scaling_type})")
    
    return ranges


def create_tuner(config_path: str = "config.yaml", tuning_config: dict = tuning_config) -> HyperparameterTuner:
    """Create the SageMaker hyperparameter tuner."""

    # create trainer
    trainer = YOLOSageMakerTrainer(config_path=config_path)
    
    # create estimator
    estimator = trainer.create_estimator(verbose=False)
    
    # Get hyperparameter ranges
    hyperparameter_ranges = get_hyperparameter_ranges_from_config(tuning_config)
    
    # Create tuner
    tuner = HyperparameterTuner(
        estimator=estimator,
        hyperparameter_ranges=hyperparameter_ranges,
        metric_definitions=estimator.metric_definitions,
        objective_type="Maximize",
        objective_metric_name="yolo:mAP_0.5",
        max_jobs=6,
        max_parallel_jobs=2,
        strategy="Hyperband",
        early_stopping_type="Auto",
        base_tuning_job_name="yolo-hpt-test",
    )
    
    print(f"Created hyperparameter tuner:")
    
    return tuner


if __name__ == "__main__":
    # ranges = get_hyperparameter_ranges_from_config(tuning_config)
    # print("Ranges: ", ranges)


    tuner = create_tuner()
    print("Tuner created: ", tuner)
    print(tuner.hyperparameter_ranges)
    print(tuner.metric_definitions)
    print(tuner.objective_type)
    print(tuner.objective_metric_name)
    print(tuner.max_jobs)
    print(tuner.max_parallel_jobs)
    print(tuner.strategy)
    print(tuner.early_stopping_type)
    print(tuner.base_tuning_job_name)

    tuner.fit(wait=False)


