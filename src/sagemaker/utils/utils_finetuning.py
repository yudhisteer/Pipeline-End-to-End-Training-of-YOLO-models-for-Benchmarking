import re
from typing import Dict, Any
from sagemaker.tuner import ContinuousParameter, CategoricalParameter, IntegerParameter


def update_yaml_hyperparams(file_path: str, lines: list, hyperparams: dict) -> None:
    """
    Update only the hyperparams section in a YAML file while preserving 
    formatting, comments, and structure. Merges new hyperparams with existing ones.
    
    Args:
        file_path: Path to the YAML file to update
        lines: List of lines from the original config file
        hyperparams: Dictionary of hyperparameters to update (will merge with existing)
    """
    # First, extract existing hyperparams to merge with new ones
    existing_hyperparams = {}
    in_hyperparams = False
    hyperparams_indent = ""
    
    # Parse existing hyperparams
    for line in lines:
        if re.match(r'^(\s*)hyperparams:\s*$', line):
            in_hyperparams = True
            hyperparams_indent = re.match(r'^(\s*)', line).group(1) + "  "
            continue
            
        if in_hyperparams:
            current_indent = re.match(r'^(\s*)', line).group(1)
            if line.strip() and len(current_indent) <= len(hyperparams_indent) - 2:
                # We've left the hyperparams section
                break
            elif line.strip() and ':' in line:
                # This is a hyperparam line
                param_line = line.strip()
                if ':' in param_line:
                    param_name = param_line.split(':')[0].strip()
                    param_value = param_line.split(':', 1)[1].strip()
                    # Try to convert to proper type
                    if param_value.lower() in ['true', 'false']:
                        existing_hyperparams[param_name] = param_value.lower() == 'true'
                    elif param_value.replace('.', '').replace('-', '').isdigit():
                        if '.' in param_value:
                            existing_hyperparams[param_name] = float(param_value)
                        else:
                            existing_hyperparams[param_name] = int(param_value)
                    else:
                        existing_hyperparams[param_name] = param_value
    
    # Merge existing with new hyperparams (new ones override existing)
    merged_hyperparams = existing_hyperparams.copy()
    merged_hyperparams.update(hyperparams)
    
    # Now rebuild the file with merged hyperparams
    new_lines = []
    in_hyperparams = False
    
    for line in lines:
        # Check if we're entering hyperparams section
        if re.match(r'^(\s*)hyperparams:\s*$', line):
            in_hyperparams = True
            hyperparams_indent = re.match(r'^(\s*)', line).group(1) + "  "
            new_lines.append(line)
            
            # Add all merged hyperparams
            for param, value in merged_hyperparams.items():
                if isinstance(value, bool):
                    formatted_value = str(value).lower()
                else:
                    formatted_value = str(value)
                new_lines.append(f"{hyperparams_indent}{param}: {formatted_value}\n")
            continue
        
        # Check if we're leaving hyperparams section
        if in_hyperparams:
            current_indent = re.match(r'^(\s*)', line).group(1)
            if line.strip() and len(current_indent) <= len(hyperparams_indent) - 2:
                in_hyperparams = False
                new_lines.append(line)
            # Skip existing hyperparams lines - we've already added the merged ones
            continue
        
        # Keep all other lines as-is
        new_lines.append(line)
    
    # Write the updated content back to the file
    with open(file_path, 'w') as f:
        f.writelines(new_lines)


def get_hyperparameter_ranges_from_config(tuning_config: dict) -> Dict:
    """
    Create SageMaker hyperparameter ranges from tuning configuration.
    
    Args:
        tuning_config: Tuning configuration dictionary from config.yaml
        
    Returns:
        Dictionary of SageMaker parameter objects
    """
    ranges = {}
    hyperparameter_ranges = tuning_config.get('hyperparameter_ranges', {})
    
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
    
    return ranges


def process_hyperparameters(best_hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process and clean hyperparameters from tuning results, converting numpy types
    to native Python types and cleaning up string values.
    
    Args:
        best_hyperparams: Raw hyperparameters from tuning job
        
    Returns:
        Processed hyperparameters with proper Python types
    """
    processed_hyperparams = {}
    
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
        
        processed_hyperparams[param] = value
    
    return processed_hyperparams
