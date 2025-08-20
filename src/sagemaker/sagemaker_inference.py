"""
SageMaker inference client for plastic bag detection model.
This module handles inference operations using existing endpoints with proper timeout handling.
"""

import boto3
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np
import sys
import os
import argparse
from typing import List, Tuple, Optional, Dict, Any
from botocore.config import Config
from botocore.exceptions import ReadTimeoutError, ConnectTimeoutError
from datetime import datetime, timedelta

import sagemaker

from utils.utils_config import (
    load_config,
    get_aws_config,
    get_inference_config
)


class SageMakerInferenceClient:
    """Handles SageMaker inference operations using existing endpoints."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize SageMaker inference client with configuration.
        
        Args:
            config: Configuration dictionary from YAML file
        """
        self.config = config or load_config()
        
        # Get AWS and inference configuration
        aws_config = get_aws_config(self.config)
        inference_config = get_inference_config(self.config)
        
        self.region = aws_config.get('region')
        
        # Set up SageMaker session and runtime with timeout configuration
        self.sess = sagemaker.Session()
        self.region = self.region or self.sess.boto_region_name
        
        # Configure boto3 client with proper timeouts
        boto_config = Config(
            region_name=self.region,
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            },
            # Increase timeouts for large images or slow models
            read_timeout=120,  # 2 minutes for reading response
            connect_timeout=60  # 1 minute for connection
        )
        
        self.runtime = boto3.client(
            service_name="runtime.sagemaker", 
            region_name=self.region,
            config=boto_config
        )
        
        # Get endpoint name from deployment config
        deployment_config = self.config.get('deployment', {})
        self.endpoint_name = deployment_config.get('endpoint_name')
        
        if not self.endpoint_name:
            raise ValueError("endpoint_name must be provided in config.yaml under 'deployment.endpoint_name'")
        
        # Store inference configuration
        self.inference_config = inference_config
        
        # Verify endpoint exists and is in service
        self._verify_endpoint()
        
        print(f"Initialized inference client for endpoint: {self.endpoint_name}")
    
    def _verify_endpoint(self):
        """Verify that the endpoint exists and is in service."""
        try:
            client = boto3.client('sagemaker', region_name=self.region)
            response = client.describe_endpoint(EndpointName=self.endpoint_name)
            status = response['EndpointStatus']
            
            if status != 'InService':
                raise ValueError(f"Endpoint {self.endpoint_name} is not in service. Status: {status}")
            
            # Check if endpoint is warming up (common cause of timeouts)
            if 'FailureReason' in response:
                print(f"Warning: Endpoint has failure reason: {response['FailureReason']}")
            
            print(f"Endpoint {self.endpoint_name} is in service")
            
            # Get endpoint configuration to check instance details
            config_name = response['EndpointConfigName']
            config_response = client.describe_endpoint_config(EndpointConfigName=config_name)
            
            for variant in config_response['ProductionVariants']:
                print(f"Instance type: {variant['InstanceType']}")
                print(f"Instance count: {variant['InitialInstanceCount']}")
                
                # Warn about cold start potential
                if variant['InitialInstanceCount'] == 1:
                    print("Warning: Single instance endpoint may experience cold starts")
            
        except client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                raise ValueError(f"Endpoint {self.endpoint_name} does not exist")
            else:
                raise ValueError(f"Error checking endpoint {self.endpoint_name}: {e}")
    
    def _optimize_image(self, image_path: str, max_size: int = 1024) -> bytes:
        """
        Optimize image size to reduce inference time and avoid timeouts.
        
        Args:
            image_path: Path to image file
            max_size: Maximum dimension for image resizing
            
        Returns:
            Optimized image as bytes
        """
        try:
            from PIL import Image
            import io
            
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                # Resize if too large
                width, height = img.size
                if max(width, height) > max_size:
                    if width > height:
                        new_width = max_size
                        new_height = int(height * max_size / width)
                    else:
                        new_height = max_size
                        new_width = int(width * max_size / height)
                    
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    print(f"Resized image from {width}x{height} to {new_width}x{new_height}")
                
                # Convert to bytes with optimized JPEG quality
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                return buffer.getvalue()
                
        except ImportError:
            print("PIL not available, using original image size")
            # Fallback to original method
            with open(image_path, "rb") as image:
                return image.read()
    
    def predict_image(self, 
        image_path: str, 
        threshold: float = None,
        max_retries: int = 3,
        optimize_image: bool = True
        ) -> List[Tuple]:
        """
        Make prediction on a single image with timeout handling.
        
        Args:
            image_path: Path to image file
            threshold: Confidence threshold for detections (uses config default if None)
            max_retries: Maximum number of retry attempts
            optimize_image: Whether to optimize image size before sending
            
        Returns:
            List of detections (class, score, x0, y0, x1, y1)
        """
        # Get prediction configuration
        prediction_config = self.inference_config.get('prediction', {})
        threshold = threshold if threshold is not None else prediction_config.get('default_threshold', 0.3)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Optimize image if requested
        if optimize_image:
            image_bytes = self._optimize_image(image_path)
        else:
            with open(image_path, "rb") as image:
                image_bytes = image.read()
        
        print(f"Image size: {len(image_bytes) / 1024:.1f} KB")
        
        # Make prediction with retries
        last_error = None
        for attempt in range(max_retries):
            try:
                print(f"Prediction attempt {attempt + 1}/{max_retries}")
                
                # Make prediction with explicit timeout handling
                endpoint_response = self.runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="image/jpeg",
                    Body=image_bytes
                )
                
                # Parse results
                results = endpoint_response["Body"].read()
                detections = json.loads(results)
                
                # Filter by threshold
                filtered_detections = []
                for det in detections['prediction']:
                    if len(det) >= 6 and det[1] >= threshold:  # score >= threshold
                        filtered_detections.append(tuple(det))
                
                print(f"Prediction successful on attempt {attempt + 1}")
                return filtered_detections
                
            except (ReadTimeoutError, ConnectTimeoutError) as e:
                last_error = e
                print(f"Timeout on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in 5 seconds...")
                    import time
                    time.sleep(5)
                else:
                    print("Max retries exceeded")
            
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
                last_error = e
                break
        
        # If all attempts failed, raise the last error
        raise Exception(f"Failed to get prediction after {max_retries} attempts. Last error: {last_error}")
    
    def check_endpoint_health(self) -> Dict[str, Any]:
        """
        Check endpoint health and provide diagnostics.
        
        Returns:
            Dictionary with health information and recommendations
        """
        health_info = {
            'status': 'unknown',
            'recommendations': [],
            'metrics': {}
        }
        
        try:
            # Check endpoint status
            client = boto3.client('sagemaker', region_name=self.region)
            response = client.describe_endpoint(EndpointName=self.endpoint_name)
            
            health_info['status'] = response['EndpointStatus']
            health_info['creation_time'] = response['CreationTime']
            
            if response['EndpointStatus'] != 'InService':
                health_info['recommendations'].append(f"Endpoint is {response['EndpointStatus']}, wait for it to be InService")
                return health_info
            
            # Check CloudWatch metrics for the endpoint
            try:
                
                
                cloudwatch = boto3.client('cloudwatch', region_name=self.region)
                
                # Get metrics for the last hour
                end_time = datetime.now(datetime.timezone.utc)
                start_time = end_time - timedelta(hours=1)
                
                metrics_to_check = [
                    'ModelLatency',
                    'OverheadLatency', 
                    'Invocations',
                    'InvocationsPer5XXErrors'
                ]
                
                for metric_name in metrics_to_check:
                    try:
                        metric_response = cloudwatch.get_metric_statistics(
                            Namespace='AWS/SageMaker',
                            MetricName=metric_name,
                            Dimensions=[
                                {
                                    'Name': 'EndpointName',
                                    'Value': self.endpoint_name
                                },
                            ],
                            StartTime=start_time,
                            EndTime=end_time,
                            Period=300,  # 5 minutes
                            Statistics=['Average', 'Maximum']
                        )
                        
                        if metric_response['Datapoints']:
                            latest = max(metric_response['Datapoints'], key=lambda x: x['Timestamp'])
                            health_info['metrics'][metric_name] = {
                                'average': latest.get('Average', 0),
                                'maximum': latest.get('Maximum', 0)
                            }
                    except Exception as e:
                        print(f"Could not get {metric_name} metric: {e}")
                
                # Analyze metrics and provide recommendations
                if 'ModelLatency' in health_info['metrics']:
                    avg_latency = health_info['metrics']['ModelLatency']['average']
                    max_latency = health_info['metrics']['ModelLatency']['maximum']
                    
                    if avg_latency > 30000:  # 30 seconds
                        health_info['recommendations'].append("High model latency detected - consider using a larger instance type")
                    if max_latency > 60000:  # 1 minute
                        health_info['recommendations'].append("Very high maximum latency - endpoint may be experiencing cold starts")
                
                if 'InvocationsPer5XXErrors' in health_info['metrics']:
                    errors = health_info['metrics']['InvocationsPer5XXErrors']['average']
                    if errors > 0:
                        health_info['recommendations'].append("5XX errors detected - check endpoint logs")
                        
            except ImportError:
                health_info['recommendations'].append("Install boto3 to get detailed metrics")
            except Exception as e:
                print(f"Could not get CloudWatch metrics: {e}")
                health_info['recommendations'].append("Could not retrieve performance metrics")
            
            # General recommendations for timeout issues
            if not health_info['recommendations']:
                health_info['recommendations'] = [
                    "Endpoint appears healthy",
                    "If experiencing timeouts, try: reducing image size, increasing client timeout, or using a larger instance type"
                ]
            
        except Exception as e:
            health_info['status'] = 'error'
            health_info['error'] = str(e)
            health_info['recommendations'].append(f"Could not check endpoint health: {e}")
        
        return health_info
    
    def create_yolo_inference_script(self) -> str:
        """
        Create a YOLO-compatible inference script for SageMaker PyTorch container.
        
        Returns:
            Inference script content as string
        """
        inference_script = '''
import torch
import json
import logging
import sys
import os
from PIL import Image
import io
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load the YOLO model for inference."""
    try:
        # Try to import ultralytics
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            raise
        
        # Look for YOLO model files
        model_files = ['best.pt', 'last.pt', 'yolo.pt']
        model_path = None
        
        for model_file in model_files:
            potential_path = os.path.join(model_dir, model_file)
            if os.path.exists(potential_path):
                model_path = potential_path
                break
        
        if not model_path:
            available_files = os.listdir(model_dir)
            logger.error(f"No YOLO model found in {model_dir}. Available files: {available_files}")
            raise FileNotFoundError("No YOLO model file found")
        
        logger.info(f"Loading YOLO model from: {model_path}")
        model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def input_fn(request_body, request_content_type):
    """Parse input data for inference."""
    try:
        if request_content_type == 'image/jpeg' or request_content_type == 'image/png':
            # Handle image input
            image = Image.open(io.BytesIO(request_body))
            return image
        elif request_content_type == 'application/json':
            # Handle JSON input (base64 encoded image)
            data = json.loads(request_body)
            if 'image' in data:
                import base64
                image_data = base64.b64decode(data['image'])
                image = Image.open(io.BytesIO(image_data))
                return image
            else:
                raise ValueError("JSON input must contain 'image' field with base64 encoded image")
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        raise

def predict_fn(input_data, model):
    """Run inference on the input data."""
    try:
        # Run YOLO inference
        results = model(input_data, verbose=False)
        
        # Extract predictions
        predictions = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Convert to normalized coordinates (0-1)
                    img_height, img_width = input_data.size[::-1] if hasattr(input_data, 'size') else (640, 640)
                    
                    predictions.append([
                        cls,           # class
                        conf,          # confidence
                        x1/img_width,  # x1 normalized
                        y1/img_height, # y1 normalized  
                        x2/img_width,  # x2 normalized
                        y2/img_height  # y2 normalized
                    ])
        
        return {"prediction": predictions}
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def output_fn(prediction, content_type):
    """Format the prediction output."""
    try:
        if content_type == 'application/json':
            return json.dumps(prediction)
        else:
            raise ValueError(f"Unsupported output content type: {content_type}")
    except Exception as e:
        logger.error(f"Error formatting output: {e}")
        raise
'''
        return inference_script
    
    def visualize_detections(self, 
        image_path: str, 
        detections: List[Tuple], 
        threshold: float = None, 
        save_path: Optional[str] = None
        ):
        """
        Visualize detections on image using configuration.
        
        Args:
            image_path: Path to image file
            detections: List of detections from predict_image()
            threshold: Confidence threshold for display (uses config default if None)
            save_path: Optional path to save visualization
        """
        # Get visualization configuration
        prediction_config = self.inference_config.get('prediction', {})
        viz_config = prediction_config.get('visualization', {})
        
        threshold = threshold if threshold is not None else prediction_config.get('default_threshold', 0.3)
        figure_size = viz_config.get('figure_size', [12, 8])
        line_width = viz_config.get('line_width', 3.5)
        font_size = viz_config.get('font_size', 12)
        dpi = viz_config.get('dpi', 150)
        
        # Load image
        img = mpimg.imread(image_path)
        plt.figure(figsize=figure_size)
        plt.imshow(img)
        
        width = img.shape[1]
        height = img.shape[0]
        colors = {}
        num_detections = 0
        
        for det in detections:
            if len(det) < 6:
                continue
                
            klass, score, x0, y0, x1, y1 = det
            
            if score < threshold:
                continue
                
            num_detections += 1
            cls_id = int(klass)
            
            # Generate random color for each class
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            
            # Convert normalized coordinates to pixel coordinates
            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)
            
            # Draw bounding box
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor=colors[cls_id],
                linewidth=line_width
            )
            plt.gca().add_patch(rect)
            
            # Add confidence score
            plt.gca().text(
                xmin,
                ymin - 2,
                f"{score:.3f}",
                bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                fontsize=font_size,
                color="white"
            )
        
        plt.title(f"Plastic Bag Detection - {num_detections} detections found")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        print(f"Number of detections: {num_detections}")
    
    def predict_and_visualize(self, 
        image_path: str, 
        threshold: float = None,
        save_path: Optional[str] = None
        ) -> List[Tuple]:
        """
        Predict and visualize detections in one step.
        
        Args:
            image_path: Path to image file
            threshold: Confidence threshold (uses config default if None)
            save_path: Optional path to save visualization
            
        Returns:
            List of detections
        """
        print(f"Processing image: {image_path}")
        
        # Make prediction
        detections = self.predict_image(image_path, threshold)
        
        # Visualize results
        self.visualize_detections(image_path, detections, threshold, save_path)
        
        return detections
    
    def get_endpoint_info(self) -> Dict[str, Any]:
        """
        Get information about the current endpoint.
        
        Returns:
            Dictionary with endpoint information
        """
        try:
            client = boto3.client('sagemaker', region_name=self.region)
            response = client.describe_endpoint(EndpointName=self.endpoint_name)
            
            info = {
                'endpoint_name': response['EndpointName'],
                'status': response['EndpointStatus'],
                'creation_time': response['CreationTime'],
                'last_modified_time': response['LastModifiedTime'],
                'endpoint_arn': response['EndpointArn']
            }
            
            # Get endpoint configuration details
            config_response = client.describe_endpoint_config(
                EndpointConfigName=response['EndpointConfigName']
            )
            
            production_variants = config_response['ProductionVariants']
            if production_variants:
                variant = production_variants[0]
                info['instance_type'] = variant['InstanceType']
                info['instance_count'] = variant['InitialInstanceCount']
                info['variant_name'] = variant['VariantName']
            
            return info
            
        except Exception as e:
            print(f"Error getting endpoint info: {e}")
            return {'error': str(e)}


def main():
    """Enhanced SageMaker inference with command-line support."""
    parser = argparse.ArgumentParser(
        description="SageMaker Inference Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check endpoint status and health
  python src/sagemaker/sagemaker_inference.py --status
  
  # Run test prediction (uses config test image)
  python src/sagemaker/sagemaker_inference.py --test
  
  # Predict on specific image
  python src/sagemaker/sagemaker_inference.py --image path/to/image.jpg
  
  # Generate YOLO inference script
  python src/sagemaker/sagemaker_inference.py --generate-script
  
  # Full run (status + health + test if configured)
  python src/sagemaker/sagemaker_inference.py
        """
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show endpoint status and info only'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test prediction using config test image'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to specific image for prediction'
    )
    
    parser.add_argument(
        '--generate-script',
        action='store_true',
        help='Generate YOLO inference script for deployment'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Confidence threshold for predictions (default: from config)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        help='Path to save visualization result'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration from: config.yaml")
    try:
        config = load_config()
        inference_config = get_inference_config(config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None
    
    # Handle generate script option (doesn't need endpoint)
    if args.generate_script:
        try:
            # Create a dummy client to access the method
            client = SageMakerInferenceClient(config=config)
            script = client.create_yolo_inference_script()
            print("Generated YOLO inference script:")
            print("=" * 50)
            print(script)
            print("=" * 50)
            return None
        except Exception as e:
            print(f"Error generating script: {e}")
            return None
    
    # Create inference client
    try:
        client = SageMakerInferenceClient(config=config)
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure the endpoint is deployed and running.")
        return None
    
    # Handle different modes
    if args.status:
        # Show status only
        show_endpoint_status(client)
        
    elif args.image:
        # Predict on specific image
        if not os.path.exists(args.image):
            print(f"Error: Image not found: {args.image}")
            return None
        
        print(f"Running prediction on: {args.image}")
        try:
            detections = client.predict_and_visualize(
                args.image, 
                threshold=args.threshold,
                save_path=args.save
            )
            print(f"Prediction completed - found {len(detections)} detections")
        except Exception as e:
            print(f"Prediction failed: {e}")
            
    elif args.test:
        # Run test prediction
        run_test_prediction(client, inference_config, args.threshold, args.save)
        
    else:
        # Full run: status + health + test (if configured)
        show_endpoint_status(client)
        run_health_check(client)
        run_test_prediction(client, inference_config, args.threshold, args.save)
    
    return client


def show_endpoint_status(client):
    """Show endpoint status and information."""
    endpoint_info = client.get_endpoint_info()
    if 'error' not in endpoint_info:
        print(f"\nEndpoint Information:")
        print(f"  Name: {endpoint_info.get('endpoint_name')}")
        print(f"  Status: {endpoint_info.get('status')}")
        print(f"  Instance Type: {endpoint_info.get('instance_type')}")
        print(f"  Instance Count: {endpoint_info.get('instance_count')}")
    else:
        print(f"Error getting endpoint info: {endpoint_info.get('error')}")


def run_health_check(client):
    """Run endpoint health check."""
    print("\nRunning endpoint health check...")
    health = client.check_endpoint_health()
    print(f"Health Status: {health['status']}")
    
    if health['recommendations']:
        print("Recommendations:")
        for rec in health['recommendations']:
            print(f"  - {rec}")
    
    if health.get('metrics'):
        print("\nRecent Metrics:")
        for metric, values in health['metrics'].items():
            print(f"  {metric}: avg={values['average']:.2f}, max={values['maximum']:.2f}")


def run_test_prediction(client, inference_config, threshold=None, save_path=None):
    """Run test prediction if configured."""
    test_config = inference_config.get('test', {})
    test_image = test_config.get('image_path')
    
    if test_image and os.path.exists(test_image):
        print(f"\nRunning test prediction on: {test_image}")
        try:
            detections = client.predict_and_visualize(
                test_image,
                threshold=threshold,
                save_path=save_path
            )
            print(f"Test completed - found {len(detections)} detections")
        except Exception as e:
            print(f"Test prediction failed: {e}")
            print("\nTroubleshooting suggestions:")
            print("1. Check if the endpoint is warmed up (try again in a few minutes)")
            print("2. Verify your image is not too large (try a smaller image)")
            print("3. Check CloudWatch logs for the endpoint")
            print("4. Consider using a larger instance type if latency is consistently high")
    elif test_image:
        print(f"Test image not found: {test_image}")
        print("Please check the 'inference.test.image_path' in your config.yaml")
    else:
        print("\nNo test image specified in config")
        print("To run a test prediction, set 'inference.test.image_path' in config.yaml")
        print("Example: inference.test.image_path: 'path/to/test/image.jpg'")
        print("Or use: python src/sagemaker/sagemaker_inference.py --image path/to/image.jpg")


if __name__ == "__main__":
    main()