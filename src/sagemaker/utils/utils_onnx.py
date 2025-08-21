#!/usr/bin/env python3
"""
Test script to validate ONNX model and perform inference.
This script checks if the ONNX model is valid and can make predictions.
"""

import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path

def test_onnx_model(onnx_path: str) -> bool:
    """
    Test if the ONNX model is valid and can perform inference.
    
    Args:
        onnx_path (str): Path to the ONNX model file
        
    Returns:
        bool: True if model is valid and can perform inference, False otherwise
    """
    print(f"Testing ONNX model: {onnx_path}")
    print("=" * 50)
    
    # Check if file exists
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found at {onnx_path}")
        return False
    
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # Convert to MB
    print(f"ONNX file found. Size: {file_size:.2f} MB")
    
    try:
        # Load and validate ONNX model
        print("\n1. Loading and validating ONNX model...")
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("ONNX model is valid")
        
        # Print model metadata
        print(f"   - IR version: {model.ir_version}")
        print(f"   - Producer: {model.producer_name}")
        print(f"   - Model version: {model.model_version}")
        print(f"   - Opset version: {model.opset_import[0].version}")
        
        # Get input and output information
        inputs = [input.name for input in model.graph.input]
        outputs = [output.name for output in model.graph.output]
        print(f"   - Inputs: {inputs}")
        print(f"   - Outputs: {outputs}")
        
        # Get input shapes
        input_shapes = []
        for input_info in model.graph.input:
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(dim.dim_value)
            input_shapes.append((input_info.name, shape))
        
        print(f"   - Input shapes: {input_shapes}")
        
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return False
    
    try:
        # Test inference with ONNX Runtime
        print("\n2. Testing inference with ONNX Runtime...")
        
        # Create inference session
        session = ort.InferenceSession(onnx_path)
        print("ONNX Runtime session created successfully")
        
        # Get input details
        input_details = session.get_inputs()
        output_details = session.get_outputs()
        
        print(f"   - Input details: {len(input_details)} inputs")
        for i, input_info in enumerate(input_details):
            print(f"     Input {i}: {input_info.name}, shape: {input_info.shape}, type: {input_info.type}")
        
        print(f"   - Output details: {len(output_details)} outputs")
        for i, output_info in enumerate(output_details):
            print(f"     Output {i}: {output_info.name}, shape: {output_info.shape}, type: {output_info.type}")
        
        # Generate sample input data
        print("\n3. Generating sample input data...")
        sample_inputs = {}
        
        for input_info in input_details:
            input_name = input_info.name
            input_shape = input_info.shape
            
            # Handle dynamic dimensions by setting them to reasonable values
            actual_shape = []
            for dim in input_shape:
                if isinstance(dim, str) or dim <= 0:
                    # Set dynamic dimensions to reasonable defaults
                    if 'batch' in str(dim).lower() or dim == 0:
                        actual_shape.append(1)  # Batch size = 1
                    elif 'height' in str(dim).lower() or dim == 0:
                        actual_shape.append(640)  # Common YOLO height
                    elif 'width' in str(dim).lower() or dim == 0:
                        actual_shape.append(640)  # Common YOLO width
                    elif 'channel' in str(dim).lower() or dim == 0:
                        actual_shape.append(3)   # RGB channels
                    else:
                        actual_shape.append(1)   # Default fallback
                else:
                    actual_shape.append(dim)
            
            # Generate random data with appropriate shape
            if input_info.type == np.float32:
                sample_data = np.random.randn(*actual_shape).astype(np.float32)
            elif input_info.type == np.float16:
                sample_data = np.random.randn(*actual_shape).astype(np.float16)
            elif input_info.type == np.int64:
                sample_data = np.random.randint(0, 1000, actual_shape, dtype=np.int64)
            else:
                sample_data = np.random.randn(*actual_shape).astype(np.float32)
            
            sample_inputs[input_name] = sample_data
            print(f"   - Generated input '{input_name}' with shape {actual_shape}")
        
        # Perform inference
        print("\n4. Performing inference...")
        outputs = session.run(None, sample_inputs)
        print("Inference completed successfully!")
        
        # Print output information
        print(f"   - Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"     Output {i}: shape {output.shape}, dtype {output.dtype}")
            if len(output.shape) > 0:
                print(f"       - Min value: {output.min():.6f}")
                print(f"       - Max value: {output.max():.6f}")
                print(f"       - Mean value: {output.mean():.6f}")
        
        return True
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return False

def main():
    """Main function to run the ONNX model test."""
    onnx_path = "models/pipelines-bpblsxjiotwz-YOLOTrainingStep-Obj-S6s6poOKsB/train/weights/best.onnx"
    
    print("ONNX Model Testing Script")
    print("=" * 50)
    
    # Test the model
    success = test_onnx_model(str(onnx_path))
    
    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: ONNX model is valid and can perform inference!")
        print("The model is ready to use.")
    else:
        print("FAILED: ONNX model has issues.")
        print("Please check the error messages above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
