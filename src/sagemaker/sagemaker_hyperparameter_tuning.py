"""
Simple POC for YOLO Hyperparameter Tuning using SageMaker.
Extends the existing YOLOSageMakerTrainer for hyperparameter optimization.
"""

import os
import pandas as pd
import yaml
from datetime import datetime
from typing import Dict, Any

from sagemaker.tuner import HyperparameterTuner
from sagemaker.parameter import (
    ContinuousParameter, 
    CategoricalParameter, 
    IntegerParameter
)

from entrypoint_trainer import YOLOSageMakerTrainer
from utils.utils_config import load_config


class YOLOHyperparameterTuner:
    """
    Simple POC for YOLO hyperparameter tuning using SageMaker.
    Focuses on key parameters: learning rate, batch size, and optimizer.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the hyperparameter tuner."""
        self.config_path = config_path
        self.config = load_config(config_path)
        self.tuning_config = self.config.get('tuning', {})
        
        # Create base trainer
        self.trainer = YOLOSageMakerTrainer(config_path=config_path)
        
        # Setup tuning job naming
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tuning_job_name = f"yolo-hpt-{self.timestamp}"
        
        print(f"Initialized YOLO Hyperparameter Tuner")
        print(f"Tuning job name: {self.tuning_job_name}")
    
    def get_hyperparameter_ranges(self) -> Dict:
        """
        Define hyperparameter ranges for tuning from config.yaml.
        """
        ranges = {}
        hyperparameter_ranges = self.tuning_config.get('hyperparameter_ranges', {})
        
        for param_name, param_config in hyperparameter_ranges.items():
            param_type = param_config.get('type')
            
            if param_type == 'continuous':
                ranges[param_name] = ContinuousParameter(
                    param_config['min'], 
                    param_config['max']
                )
            elif param_type == 'categorical':
                ranges[param_name] = CategoricalParameter(param_config['values'])
            elif param_type == 'integer':
                ranges[param_name] = IntegerParameter(
                    param_config['min'], 
                    param_config['max']
                )
            else:
                print(f"Warning: Unknown parameter type '{param_type}' for {param_name}")
        
        print("Hyperparameter ranges for tuning (from config):")
        for param, range_def in ranges.items():
            if isinstance(range_def, ContinuousParameter):
                print(f"  {param}: continuous [{range_def.min_value} - {range_def.max_value}]")
            elif isinstance(range_def, CategoricalParameter):
                print(f"  {param}: categorical {range_def.values}")
            elif isinstance(range_def, IntegerParameter):
                print(f"  {param}: integer [{range_def.min_value} - {range_def.max_value}]")
            else:
                print(f"  {param}: {range_def}")
        
        if not ranges:
            print("Warning: No hyperparameter ranges found in config.yaml")
            print("Please configure 'tuning.hyperparameter_ranges' section")
        
        return ranges
    
    def create_tuner(self) -> HyperparameterTuner:
        """Create the SageMaker hyperparameter tuner."""
        
        # Create estimator
        estimator = self.trainer.create_estimator()
        
        # Get hyperparameter ranges
        hyperparameter_ranges = self.get_hyperparameter_ranges()
        
        # Configure tuning job from config
        max_jobs = self.tuning_config.get('max_jobs', 6)
        max_parallel_jobs = self.tuning_config.get('max_parallel_jobs', 2)
        objective_metric = f"yolo:{self.tuning_config.get('objective_metric', 'mAP_0.5')}"
        objective_type = self.tuning_config.get('objective_type', 'Maximize')
        
        # Create tuner
        tuner = HyperparameterTuner(
            estimator=estimator,
            objective_metric_name=objective_metric,
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=estimator.metric_definitions,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            objective_type=objective_type,
            base_tuning_job_name=self.tuning_job_name
        )
        
        print(f"Created hyperparameter tuner:")
        print(f"  Max jobs: {max_jobs}")
        print(f"  Max parallel jobs: {max_parallel_jobs}")
        print(f"  Objective metric: {objective_metric}")
        print(f"  Objective type: {objective_type}")
        
        return tuner
    
    def start_tuning_job(self, wait: bool = False) -> HyperparameterTuner:
        """
        Start the hyperparameter tuning job.
        
        Args:
            wait: Whether to wait for completion
            
        Returns:
            The tuner object
        """
        # Create tuner
        tuner = self.create_tuner()
        
        # Prepare training inputs
        inputs = self.trainer.prepare_training_inputs()
        
        print(f"Starting hyperparameter tuning job...")
        print(f"  Job name: {self.tuning_job_name}")
        print(f"  Training data: {self.trainer.s3_training_data}")
        
        # Start tuning
        tuner.fit(inputs, wait=wait)
        
        if wait:
            print(f"Hyperparameter tuning completed!")
            self._display_best_training_job(tuner)
        else:
            print(f"Hyperparameter tuning job started.")
            print(f"Monitor progress in SageMaker console:")
            print(f"  https://{self.trainer.region}.console.aws.amazon.com/sagemaker/home?region={self.trainer.region}#/hyper-tuning-jobs/{self.tuning_job_name}")
        
        return tuner
    
    def _display_best_training_job(self, tuner: HyperparameterTuner):
        """Display comprehensive information about the hyperparameter tuning results."""
        try:
            print("\n" + "="*60)
            print("HYPERPARAMETER TUNING RESULTS")
            print("="*60)
            
            # Get analytics
            analytics = tuner.analytics()
            if not hasattr(analytics, 'dataframe'):
                print("Analytics dataframe not available")
                return
                
            df = analytics.dataframe()
            if df.empty:
                print("No results found!")
                return
            
            # Sort by objective value (mAP@0.5)
            df_sorted = df.sort_values('FinalObjectiveValue', ascending=False)
            
            # Summary statistics
            print(f"Tuning Job Summary:")
            print(f"  Total jobs: {len(df)}")
            print(f"  Completed jobs: {len(df[df['TrainingJobStatus'] == 'Completed'])}")
            print(f"  Failed jobs: {len(df[df['TrainingJobStatus'] == 'Failed'])}")
            
            completed_df = df[df['TrainingJobStatus'] == 'Completed']
            if not completed_df.empty:
                print(f"  Best mAP@0.5: {completed_df['FinalObjectiveValue'].max():.4f}")
                print(f"  Worst mAP@0.5: {completed_df['FinalObjectiveValue'].min():.4f}")
                print(f"  Average mAP@0.5: {completed_df['FinalObjectiveValue'].mean():.4f}")
                
                # Check if all values are the same
                unique_scores = completed_df['FinalObjectiveValue'].nunique()
                if unique_scores == 1:
                    print(f"  WARNING: All jobs achieved the same mAP@0.5 score!")
                    print(f"      This suggests hyperparameters may not be having an impact.")
                    print(f"      Consider: longer training time, different hyperparameters, or larger dataset.")
            
            # Show top 3 results
            print(f"\nTop 3 Results:")
            print("-" * 50)
            
            for i, (idx, row) in enumerate(df_sorted.head(3).iterrows()):
                if row['TrainingJobStatus'] == 'Completed':
                    print(f"{i+1}. Training Job: {row['TrainingJobName']}")
                    print(f"   mAP@0.5: {row['FinalObjectiveValue']:.4f}")
                    print(f"   Status: {row['TrainingJobStatus']}")
                    
                    # Show hyperparameters
                    print(f"   Hyperparameters:")
                    hyperparam_found = False
                    for col in df.columns:
                        # Check for hyperparameter columns
                        if col in ['batch', 'lr0', 'optimizer', 'epochs', 'imgsz'] or col.startswith('hp_'):
                            param_name = col.replace('hp_', '')
                            print(f"     {param_name}: {row[col]}")
                            hyperparam_found = True
                    
                    if not hyperparam_found:
                        print(f"     No hyperparameter columns found")
                    print()
            
            # Hyperparameter impact analysis
            print(f"Hyperparameter Impact Analysis:")
            print("-" * 50)
            
            if not completed_df.empty:
                print("Impact on mAP@0.5:")
                hyperparam_cols = []
                for col in completed_df.columns:
                    if col in ['batch', 'lr0', 'optimizer', 'epochs', 'imgsz'] or col.startswith('hp_'):
                        hyperparam_cols.append(col)
                        param_name = col.replace('hp_', '')
                        
                        # Only calculate correlation for numeric columns
                        if completed_df[col].dtype in ['int64', 'float64']:
                            try:
                                # Check if there's variation in the data
                                if completed_df[col].nunique() <= 1:
                                    print(f"  {param_name}: no variation (all values same)")
                                elif completed_df['FinalObjectiveValue'].nunique() <= 1:
                                    print(f"  {param_name}: no variation in mAP@0.5 (all scores same)")
                                else:
                                    # Suppress numpy warnings for this calculation
                                    import warnings
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        correlation = completed_df[col].corr(completed_df['FinalObjectiveValue'])
                                        if not pd.isna(correlation):
                                            print(f"  {param_name}: correlation = {correlation:.3f}")
                                        else:
                                            print(f"  {param_name}: unable to calculate correlation")
                            except Exception as e:
                                print(f"  {param_name}: error calculating correlation")
                        else:
                            # For categorical parameters, show value distribution
                            print(f"  {param_name}: (categorical)")
                            try:
                                value_performance = completed_df.groupby(col)['FinalObjectiveValue'].mean().sort_values(ascending=False)
                                for value, perf in value_performance.items():
                                    print(f"    {value}: avg mAP@0.5 = {perf:.4f}")
                            except:
                                print(f"    Error analyzing {param_name}")
                
                if not hyperparam_cols:
                    print("  No hyperparameter columns detected for analysis")
            
            # Best hyperparameters recommendation
            if not completed_df.empty:
                best_row = completed_df.loc[completed_df['FinalObjectiveValue'].idxmax()]
                print(f"\nRECOMMENDED HYPERPARAMETERS:")
                print("-" * 50)
                print(f"Based on best result (mAP@0.5 = {best_row['FinalObjectiveValue']:.4f}):")
                
                recommended_params = {}
                for col in completed_df.columns:
                    if col in ['batch', 'lr0', 'optimizer', 'epochs', 'imgsz'] or col.startswith('hp_'):
                        param_name = col.replace('hp_', '')
                        recommended_params[param_name] = best_row[col]
                        print(f"  {param_name}: {best_row[col]}")
                
                if recommended_params:
                    print(f"\nTo use these hyperparameters, update your config.yaml:")
                    print("training:")
                    print("  hyperparams:")
                    for param, value in recommended_params.items():
                        if isinstance(value, str):
                            print(f"    {param}: {value}")
                        else:
                            print(f"    {param}: {value}")
                else:
                    print("  No hyperparameter data found in results")
            
            # Auto-update config if enabled and we have hyperparameters
            if (self.tuning_config.get('update_config', False) and 
                not completed_df.empty and recommended_params):
                print(f"\nAuto-update enabled - updating config.yaml...")
                if self.update_config_with_best_hyperparameters(recommended_params):
                    print(f"Config updated! Ready for next training run with optimized hyperparameters.")
                else:
                    print(f"Config update failed. You can manually copy the recommended hyperparameters.")
            elif self.tuning_config.get('update_config', False):
                print(f"\nAuto-update enabled but no hyperparameters found to update")
            
            print("="*60)
            
        except Exception as e:
            print(f"Error displaying tuning results: {e}")
            print("You can view detailed results in the SageMaker console")
            import traceback
            traceback.print_exc()
    
    def update_config_with_best_hyperparameters(self, best_hyperparams: Dict[str, Any]) -> bool:
        """
        Update config.yaml with the best hyperparameters from tuning.
        
        Args:
            best_hyperparams: Dictionary of best hyperparameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\n" + "="*50)
            print("UPDATING CONFIG.YAML WITH BEST HYPERPARAMETERS")
            print("="*50)
            
            # Read current config.yaml
            with open(self.config_path, 'r') as f:
                config_content = f.read()
            
            # Parse as YAML
            config_data = yaml.safe_load(config_content)
            
            # Backup original hyperparams
            original_hyperparams = config_data.get('training', {}).get('hyperparams', {}).copy()
            
            print(f"Original hyperparameters:")
            for key, value in original_hyperparams.items():
                print(f"  {key}: {value}")
            
            # Update with best hyperparameters
            if 'training' not in config_data:
                config_data['training'] = {}
            if 'hyperparams' not in config_data['training']:
                config_data['training']['hyperparams'] = {}
            
            print(f"\nUpdating with best hyperparameters:")
            for param, value in best_hyperparams.items():
                # Convert all values to proper Python types (not numpy)
                if hasattr(value, 'item'):  # numpy scalar
                    value = value.item()
                elif isinstance(value, str):
                    # Clean up quoted strings and try to convert numbers
                    value = value.strip('"\'')
                    try:
                        # Try to convert to int first, then float
                        if value.replace('.', '').replace('-', '').isdigit():
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                    except ValueError:
                        # Keep as string for categorical values
                        pass
                
                # Ensure we have basic Python types
                if hasattr(value, 'dtype'):  # Any remaining numpy types
                    value = value.item()
                
                config_data['training']['hyperparams'][param] = value
                print(f"  {param}: {original_hyperparams.get(param, 'not set')} → {value}")
            
            # Write updated config
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, 
                         default_flow_style=False, 
                         sort_keys=False,
                         )
            
            print(f"Successfully updated {self.config_path}")
            print(f"   Use these hyperparameters for your next training run!")
            print("="*50)
            
            return True
            
        except Exception as e:
            print(f"Error updating config.yaml: {e}")
            print(f"   Original config preserved")
            return False
    
    def run_tuning(self, wait_for_completion: bool = True) -> Dict[str, Any]:
        """
        Complete tuning workflow.
        
        Args:
            wait_for_completion: Whether to wait for completion
            
        Returns:
            Tuning job information
        """
        print("="*60)
        print("Starting YOLO Hyperparameter Tuning POC")
        print("="*60)
        
        # Start tuning job
        tuner = self.start_tuning_job(wait=wait_for_completion)
        
        # Only show completion message if we waited for completion
        if wait_for_completion:
            print("\n" + "="*60)
            print("YOLO Hyperparameter Tuning COMPLETED")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("YOLO Hyperparameter Tuning STARTED")
            print("="*60)
        
        return {
            "tuning_job_name": self.tuning_job_name,
            "timestamp": self.timestamp,
            "max_jobs": self.tuning_config.get('max_jobs', 6),
            "status": "completed" if wait_for_completion else "running"
        }


def main():
    """Main function for standalone execution."""
    config_path = os.environ.get("YOLO_CONFIG_PATH", "config.yaml")
    wait_for_completion = os.environ.get("YOLO_WAIT", "true").lower() == "true"
    
    try:
        print(f"Loading configuration from: {config_path}")
        tuner = YOLOHyperparameterTuner(config_path=config_path)
        
        result = tuner.run_tuning(wait_for_completion=wait_for_completion)
        
        print(f"\nTuning Results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"Error running hyperparameter tuning: {str(e)}")
        raise


if __name__ == "__main__":
    main()


# Example usage:
"""
Simple POC for YOLO hyperparameter tuning.

Usage Examples:

1. Run tuning job (basic):
python src/sagemaker/sagemaker_hyperparameter_tuning.py

2. Run and wait for completion:
export YOLO_WAIT=true
python src/sagemaker/sagemaker_hyperparameter_tuning.py

3. Programmatic usage:
from src.sagemaker.sagemaker_hyperparameter_tuning import YOLOHyperparameterTuner

tuner = YOLOHyperparameterTuner("config.yaml")
result = tuner.run_tuning(wait_for_completion=False)

The POC tunes hyperparameters defined in config.yaml under tuning.hyperparameter_ranges.
Results are optimized for the specified objective metric (default: mAP@0.5).
"""